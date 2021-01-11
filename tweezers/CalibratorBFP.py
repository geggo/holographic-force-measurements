# -*- coding: utf-8 -*-

from __future__ import print_function
#from traits.etsconfig.api import ETSConfig
#ETSConfig.toolkit = 'qt4'

import threading
import timeit
import copy

import numpy as np
import scipy.ndimage
import scipy.signal
import scipy.spatial.distance
import skimage.feature
import skimage.transform
from traits.api import (HasTraits, Int, Float, Str, Range, Python,
                        Bool, Event, Button,
                        Enum, Array, List, Dict, Tuple,
                        DelegatesTo, Instance, Property,
                        property_depends_on
                        )
from traits_persistence import HasTraitsPersistent
from traitsui.api import (View, Item, Group, HGroup,
                          VGroup, HSplit, VSplit,
                          TextEditor, ListEditor, EnumEditor,
                          Handler)

from camimage import CamImage
from glimgwidget import (GLImageWithOverlaysEditor, OverlayTrait,
                         OverlayTraitCircle,
                         OverlayTraitGrid, OverlayTraitWarpArrows)

import pyopencl as cl
import pyopencl.array as cla

cl_platform = cl.get_platforms()[0]
cl_device = cl_platform.get_devices(cl.device_type.GPU)[0]
cl_context = cl.Context([cl_device])
cl_queue = cl.CommandQueue(cl_context)


class MarkerLocalizer(HasTraitsPersistent):
    settings_id = 'marker_localizer'


class MarkerLocalizerThreeCircles(MarkerLocalizer):
    """
    Detect position of three circles in image (via hough_circle transform),
    sort to form a oriented leg
    """

    # threshold used to make image binary for hough transformation
    binarization_threshold = Float(0.2)

    # lower/upper limit of radii in hough transformation (in px)
    radius_min = Float(20)
    radius_max = Float(40)
    radius_step_size = Float(1.)

    # erase (set to zero) ring at detected circle
    erase_width = Float(9.)

    # results
    #center = Tuple(Float, Float, value=(np.nan, np.nan), transient=True) # position central circle xy
    _centers = Array(value=np.NaN*np.zeros((3,2)), transient=True)  #position circle centers xy, relative ROI origin
    _radius  = Float
    _offset_roi = Array(value=np.zeros((2,)))

    center = Property
    grid_vectors = Property #estimated grid vectors


    traits_view = View(
        HGroup(
            Item('radius_min'),
            Item('radius_max'),
            Item('radius_step_size'),
        ),
        Item('binarization_threshold'),
        Item('erase_width'),

        HGroup(
            Item('center'),
            Item('grid_vectors'),
            Item('_radius'),
            style='readonly',
        )
    )

    def localize(self, image):
        """ find center

        uses circular hough transformation to determine center and radius.

        parameters:
        ----------
        image: Image
            original image
        :type image: CamImage

        returns:
        -------
        center: array (2, )
            center position (x,y) of circle (in px) in image coordinates

        """
        data_roi = image.data_roi

        # range of circle radii for the hough transformation
        hough_radii = np.arange(self.radius_min,
                                self.radius_max,
                                self.radius_step_size)

        ## hough_radii = np.array([22,])

        # binarize image for hough transformation
        image_sel_binarized = data_roi > (self.binarization_threshold * 2**image.bits)

        # perform hough circle transformation
        hough_res = skimage.transform.hough_circle(image_sel_binarized, hough_radii)

        #find best radius
        rad_idx, center_row, center_col = np.unravel_index(np.argmax(hough_res), hough_res.shape)
        radius = hough_radii[rad_idx]
        
        # find three circles (with radius selected above)
        accum, cx, cy, rad = skimage.transform.hough_circle_peaks(hough_res[rad_idx:rad_idx+1], 
                                                                  hough_radii[rad_idx:rad_idx+1],
                                                                  min_xdistance = 10,
                                                                  min_ydistance = 10,
                                                                  num_peaks = 3, #number circles per radius
                                                                  )
        centers = np.array([cx, cy]).T

        # calculate pairwise distances 01, 02, 12
        #find longest edge of triangle: -> opposite edge is rectangular corner
        v0 = 2 - np.argmax( scipy.spatial.distance.pdist(centers) )
        c0 = centers[v0]

        #find positive oriented leg
        v1, v2 = {0: (2,1),
                  1: (2,0),
                  2: (0,1)}[v0]
        if np.cross(centers[v1]-c0, centers[v2]-c0) < 0:
            c1, c2 = centers[v1], centers[v2]
        else:
            c1, c2 = centers[v2], centers[v1]
            
        # mirrored image: flip markers
        c1, c2 = c2, c1

        #store localizers data (positions (x,y) relative to image.pos)
        self._offset_roi = np.array(image.pos)
        self._centers = np.array([c0, c1, c2])
        self._radius  = radius

        ##return center in image coordinates
        #return c0 + np.array(image.pos)
        return self.center
        
    @property_depends_on('_centers')
    def _get_grid_vectors(self):
        # return grid unit vectors (gx: array shape (2,), gy) estimated from localizers
        c0, c1, c2 = self._centers
        scale = 1./4 #TODO: scale depends on localizer position
        return scale*(c1-c0), scale*(c2-c0)

    @property_depends_on('_centers,_offset_roi')
    def _get_center(self):
        return self._centers[0] + self._offset_roi
    

    def mask_localizers(self, image):
        """
        mask localizer circles in image
        :param image:
        :type image: CamImage
        """

        #remove circle from data_roi
        ##data_roi = image.data_roi.copy() #do not modify original data_roi

        image = copy.deepcopy(image)
        
        data_roi = image.data_roi

        radius = self._radius        
        for center in self._centers:
            #circle center in data coords
            cc, cr = center
            sel1 = slice(int(cr - 2*radius), int(cr + 2*radius)) #TODO: radius + 0.5 erase_width + margin
            sel2 = slice(int(cc - 2*radius), int(cc + 2*radius))
            data_sel = data_roi[sel1, sel2]
            X, Y = np.mgrid[sel1, sel2]
            R2 = np.square(X-cr) + np.square(Y-cc)
            ring = ((radius - 0.5*self.erase_width)**2 < R2) & \
                (R2 < (radius + 0.5*self.erase_width)**2)
            data_sel[ring] = 0
        
        return image

class GPUWarp(HasTraits):
    #settings_id = 'gpuwarp'
    shape = (1024, 1024) #(512, 512)
    warp_coords = Array( shape = shape+(2,), dtype = np.float32)
    warp_scaling = Array(shape=shape, dtype=np.float32)
    _warp_coords_buf = Python  #cl.Buffer
    _img_warped_buf = Python  #cl.Buffer
    kernel = Python  #cl.Kernel

    src_shape = (0,0)  # Tuple(Int, Int)
    src_offset = (0,0)  # Tuple(CFloat, CFloat)
    src_bits = 8


    kernel_src = """
    __constant sampler_t sampler_img_src = (CLK_NORMALIZED_COORDS_TRUE
                                            | CLK_ADDRESS_CLAMP 
                                            | CLK_FILTER_LINEAR
                                            );

    __kernel
    void warp(
    __global float2* warp_coords,
    __global float* warp_scaling,
    float2 warp_coords_offset,
    float2 warp_coords_scaling,
    __read_only image2d_t img_src,
    //__global uchar* img_dst
    __global ushort* img_dst
    )
    {
    const unsigned int xid = get_global_id(0);
    const unsigned int yid = get_global_id(1);
    const unsigned int id = xid + yid*get_global_size(0);
    
    const float2 pos_src = (warp_coords[id] - warp_coords_offset + 0.5f)*warp_coords_scaling;
    const float scale = warp_scaling[id];
    const float dst = clamp(scale * read_imagef(img_src, sampler_img_src, pos_src).x,
                        0.f,
                        1.f);
    //img_dst[id] = convert_uchar_sat(dst*255.f);
    img_dst[id] = convert_ushort_sat(dst*65535.f);
    }
    
"""

    def __init__(self, *args, **kwargs):
        super(GPUWarp, self).__init__(*args, **kwargs)
        self._warp_coords_buf = cla.zeros(cl_queue, self.shape + (2,), dtype=np.float32)
        self._warp_scaling_buf = cla.zeros(cl_queue, self.shape, dtype=np.float32)
        self._img_warped_buf = cla.zeros(cl_queue, self.shape, dtype=np.uint16)

        #init cl
        program = cl.Program(cl_context, self.kernel_src).build()
        #build_info = program.get_build_info(self.device, cl.program_build_info.LOG)
        #if build_info:
        #    print 'build info GPUWarp:'
        #    print build_info
        self.kernel = program.warp

    def set_source_image(self, src):
        src_shape = (src.roi_h, src.roi_w) #src.data_roi.shape
        src_offset = src.pos

        if self.src_offset != src_offset:
            self.src_offset = src_offset

        if self.src_shape != src_shape or self.src_bits != src.bits:  # size changed, reinit source image buffer
            self.src_shape = src_shape
            self.src_bits = src.bits
            self._img_src_buf = cl.Image(cl_context,
                                         cl.mem_flags.HOST_WRITE_ONLY | cl.mem_flags.READ_ONLY,
                                         format = cl.ImageFormat(cl.channel_order.R,
                                                                 {8: cl.channel_type.UNORM_INT8,
                                                                 16: cl.channel_type.UNORM_INT16}[src.bits],
                                                                 ),
                                         shape = (src_shape[1], src_shape[0])
                                         #is_array=False,
            )
            
        cl.enqueue_copy(cl_queue,
                        self._img_src_buf,
                        src.data_roi,
                        origin = (0,0),
                        region = (src_shape[1], src_shape[0]),
                        )


    def _warp_coords_changed(self, value):
        self._warp_coords_buf.set(value)
        print('updated warp coords')

    def _warp_scaling_changed(self, value):
        self._warp_scaling_buf.set(value)

    def warp_image(self):
        self.kernel.set_args(
            self._warp_coords_buf.data,
            self._warp_scaling_buf.data,
            cl.cltypes.make_float2(*self.src_offset),
            cl.cltypes.make_float2(1. / self.src_shape[1],
                                   1. / self.src_shape[0]),
            self._img_src_buf,
            self._img_warped_buf.data,
            )
        event = cl.enqueue_nd_range_kernel(cl_queue,
                                           self.kernel,
                                           self.shape,
                                           (8, 8),
                                           )
        event.wait()

    
class CalibratorBFPGrid(HasTraitsPersistent):
    settings_id = 'calibratorbfp-grid'
    parent = Instance(HasTraits, transient=True) # reference to main controller
    img = DelegatesTo('parent', prefix='image_bfp') # bfp image (camimage type)

    calculate = Button
    _calculation_thread = Instance(threading.Thread, transient=True)
    _busy = Bool(False, transient=True)

    marker_localizer = Instance(MarkerLocalizerThreeCircles,  # TODO should be MarkerLocalizer, accept subclass?
                                MarkerLocalizerThreeCircles(),
                                )

    gpuwarper = Instance(GPUWarp, GPUWarp(), transient=True)

    normalize = Bool(False)
    calibrate_normalization = Button
    warp_scaling_geometry = Array(shape=GPUWarp.shape, dtype=np.float32, transient=True)
    warp_scaling_flatfield = Array(shape=GPUWarp.shape, dtype=np.float32, transient=True)

    traits_view = View(
        Item('calculate', show_label=False, resizable=True,
             ),
        VGroup(
            Item('marker_localizer', style='custom', show_label=False),
            label='marker localizer', show_border=True,
            ),
        HGroup(
            Item('calibrate_normalization'),
            Item('normalize'),
        ),
        resizable=True,
        title='CalibratorBFPGrid',
        )

    def __init__(self, *args, **kwargs):
        super(CalibratorBFPGrid, self).__init__(*args, **kwargs)

        self.on_trait_change(self.on_update,
                             'calculate', #img,calculate
                             dispatch='ui')

        self.on_trait_change(self._update_warped_image,
                             'img',
                             dispatch='ui',
        )
        

    def on_update(self):
        if self._calculation_thread is not None and self._calculation_thread.is_alive():
            print('busy')
            return
        
        self._calculation_thread = t = threading.Thread(target=self._calculate)
        t.start()

    def _calculate(self):
        tic = timeit.default_timer()
        
        # detect localizers
        center = self.marker_localizer.localize(self.img)
        # self.parent.overlay_bfp_circle_center.center = (int(center[0]), int(center[1]))

        self.parent.overlay_bfp_warp_arrows.grid_origin = tuple(center)
        self.parent.overlay_bfp_warp_arrows.grid_lattice_vector1 = tuple(self.marker_localizer.grid_vectors[0])
        self.parent.overlay_bfp_warp_arrows.grid_lattice_vector2 = tuple(self.marker_localizer.grid_vectors[1])
        self.parent.overlay_bfp_warp_arrows.displacements = np.zeros((17,17,2), np.float32)

        # mask localizers
        img_masked = self.marker_localizer.mask_localizers(self.img)
        #self.parent.debug_img = img_masked

        ## testing: smoothed image
        # convolve with gaussian
        data_roi_smooth = self._convolve_data(img_masked.data_roi)

        ### testing: original data
        #data_roi_smooth = img_masked.data_roi

        #self.parent.debug_img.data_roi = data_roi_smooth
        #self.parent.debug_img = self.parent.debug_img #trigger update

        
        # detect local maxima, refine position maxima via quadratic fix
        spots = self._spot_positions(data_roi_smooth)        
        
        #self.dbgimg1 = data_roi_smooth
        #self.dbgimg1 = img_masked.data_roi
        #self.dbgimg1 = self.img.data_roi
        #self.dbgdata1 = spots.copy()
        
        spots += np.array(self.img.pos) #to image coordinates

        spots_gridded, deviation = self._assign_spots_to_grid(spots)
        deviation = deviation.astype(np.float32)
        
        self.parent.overlay_bfp_warp_arrows.displacements = -deviation

        self._find_warp()
        self._prepare_warp_image_data()
        
        self._update_warped_image()
        
        print("time elapsed %.0fms"% (1000 * (timeit.default_timer() - tic)))

    def _update_warped_image(self):
        #img_warp = self._warp_image(self.img)
        img_warp = self._warp_image_gpu(self.img)
        self.parent.image_bfp_warped = img_warp


    def _convolve_data(self, data):
        # convolve data with gaussian
        # TODO: make size/width an attribute of class
        gauss1d = scipy.signal.gaussian(M=9, std=2, sym=True) #width M should be odd?
        gauss2d = np.outer(gauss1d, gauss1d)
        gauss2d *= 1./ np.sum(gauss2d)
        return scipy.signal.fftconvolve(data, gauss2d, mode='same')
        
    def _spot_positions(self, data):
        """determine spot positions

        returns: array (N_peaks, 2) of spots (local maxima) as (x, y), relative ROI origin
        """

        # get position of local maxima with minimum distance apart
        peaks_idx = skimage.feature.peak_local_max(data,
                                               num_peaks=200, #TODO: make number of peaks an attribute
                                               min_distance=10, #TODO: make min_distance an attribute/argument
                                               exclude_border=True,
                                               )
                
        #refine peaks via quadratic fit
        x, y = np.meshgrid(np.arange(3)-3//2, #fit size, centered
                           np.arange(3)-3//2,
                           #indexing='ij', 
                           sparse=False) #NOTE: xy indexing

        D = np.stack( (x, y, x*x + y*y, np.ones_like(x)) )
        D.shape = (4,-1)

        peaks = np.zeros_like(peaks_idx, dtype=np.float64)
        for k, p_idx in enumerate(peaks_idx):
            sel1 = slice(p_idx[0]-1, p_idx[0]+1+1) #TODO: 3//2
            sel2 = slice(p_idx[1]-1, p_idx[1]+1+1)
            data_sel = data[sel1, sel2]
            
            b = np.linalg.lstsq(D.T, data_sel.flatten())[0]
            p = -0.5*b[:2]/b[2] #TODO: NaN?
            p = np.clip(p, -1, 1)
            
            peaks[k] = p + p_idx[[1,0]]
        
        return peaks # list of (x_peak, y_peak)

    def __find_closest_spot(self, spots, pos, max_dist=15):
        distance = np.sqrt(np.square(spots-pos).sum(axis=-1))
        idx = np.argmin(distance)
        closest_spot = spots[idx]
        return closest_spot if distance[idx] < max_dist else closest_spot*np.NaN 

    def _assign_spots_to_grid(self, spots):
        #spots: array (Nspots, 2), in image coordinates (xy)!

        center = self.marker_localizer.center
        peaks_G = np.stack( self.marker_localizer.grid_vectors )
                
        g1, g2 = self.marker_localizer.grid_vectors
        g2 = -g2 #TODO: not a left handed leg in image coordinates :-(

        spots_grid = np.zeros((17,17,2))
        spots_estimated = np.zeros((17,17,2))
        for k in range(17):
            for l in range(17):
                spots_estimated[k, l] = estimated =  (l-8)*g1 + (k-8)*g2 + center #!!!!!
                spots_grid[k, l] = self.__find_closest_spot(spots, estimated)

        deviation = spots_grid - spots_estimated
        
        #print spots_estimated[4,5]
        #deviation[4,5] = np.NaN #mark 4,4 spot
        
        self.spots_grid = spots_grid #(17, 17, 2) array of spots positions, NaN if no spot present

        return spots_estimated, deviation


    def _find_warp(self):
        M = self.spots_grid.shape[0]
        
        # spot positions ideal
        g = (np.arange(M) - M//2) # -8, ..., 0, ..., 8
        spots_ideal = np.stack(np.meshgrid(g, g, sparse=False)).reshape((2,-1)).T #(17*17, 2)
        
        # spot positions real
        spots_real = self.spots_grid.reshape((-1,2))

        ok = np.isfinite(spots_real[:,0]) & np.isfinite(spots_real[:,1])

        transform = skimage.transform.PolynomialTransform()
        transform.estimate(spots_ideal[ok], spots_real[ok],
                           order = 4, #TODO: make order a param
                           )

        self.transform = transform

    def _prepare_warp_image_data(self):
        warper = self.gpuwarper

        #N = 512
        N, M = warper.shape

        x = ((np.arange(N)+0.5)*1./M - 0.5)*16.
        y = ((np.arange(N)+0.5)*1./N - 0.5)*16.

        X,Y = np.meshgrid(x, y, sparse=False, indexing='xy')
        xy = np.empty( (M*N,2) )
        xy[:,0] = X.ravel()
        xy[:,1] = Y.ravel()
        
        txy = self.transform(xy)
        warper.warp_coords = txy.reshape((N,M,2)).astype(np.float32)

        d = np.gradient(txy.reshape((N,M,2)), axis=(1,0))
        warp_scaling = np.cross(d[0], d[1], axis=-1)
        warp_scaling *= 1./warp_scaling[N//2, M//2]  # adjust scale to 1 at center (hack)
        self.warp_scaling_geometry = warp_scaling.astype(np.float32)
        self.warp_scaling_flatfield.fill(1.0)
        self._normalize_changed(False)  # force self.normalize = False

        #warper.warp_scaling = self.warp_scaling_geometry.astype(np.float32)  # obsolete, done in normalize switch normalize off

    def _warp_image_gpu(self, img):
        warper = self.gpuwarper
        warper.set_source_image(img)
        warper.warp_image()
        img_warp_data = warper._img_warped_buf.get() # 3 ms
        #return CamImage(data=(img_warp_data*{8: 255, 16: 65535}[img.bits]).astype(img.dtype)) #slow 4ms!?
        warped_image = CamImage(data=img_warp_data)
        warped_image.timestamp = img.timestamp
        warped_image.frameNr = img.frameNr
        return warped_image

    def _calibrate_normalization_changed(self):
        normalize = self.normalize
        self.normalize = False
        self._update_warped_image()
        #img_warp = self._warp_image_gpu(self.img)
        img_warp = self.parent.image_bfp_warped.data_fullsize
        scaling = 2**self.img.bits/2./img_warp  # TODO: make target value an attribute
        scaling[np.isnan(scaling)] = 0.
        scaling[scaling>5.] = 5.
        self.warp_scaling_flatfield = scaling
        self.normalize = normalize

    def _normalize_changed(self, normalize):
        if normalize:
            self.gpuwarper.warp_scaling = self.warp_scaling_geometry * self.warp_scaling_flatfield
        else:
            self.gpuwarper.warp_scaling = self.warp_scaling_geometry
        self._update_warped_image()


def test_image(name, intensity_scale=1, dtype=np.uint8):
    # type: (str, int, object) -> camimage.CamImage
    """return image file from disk as camimage.CamImage"""
    import camimage
    from PIL import Image
    import os.path
    filename = os.path.join(os.path.split(__file__)[0], name)
    data = np.asarray(Image.open(filename), dtype=dtype) * intensity_scale
    img = camimage.CamImage(w=1920, h=1200, data=data)
    img.pos = (200, 100)
    return img

class TestController(Handler, HasTraitsPersistent):
    """stripped down version of MainController (holo.py) for testing of Calibrator"""

    settings_id = 'test-controller-calibrator-bfp'
    image_bfp = Instance(CamImage, comparison_mode=0, transient=True)
    image_bfp_warped = Instance(CamImage, CamImage(w=1920, h=1200),
                                comparison_mode=0, transient=True)

    _test_images = Dict(transient=True)
    _test_images_names = Property(List, depends_on='_test_images')
    test_image_name = Str()

    def __test_images_default(self):
        d = {'black': CamImage(w=1920, h=1200),
             'dots': test_image('CalibratorBFP test data/dots.png',
                                intensity_scale=1, dtype=np.uint16),
             'lines': test_image('CalibratorBFP test data/lines.png',
                                 intensity_scale=1, dtype=np.uint16),
             'traps': test_image('CalibratorBFP test data/traps.png',
                                 intensity_scale=1, dtype=np.uint16),
             'test pattern': test_image('CalibratorBFP test data/test_pattern.png'),
             }
        return d

    def _get__test_images_names(self):
        return sorted(list(self._test_images.keys()))

    def _test_image_name_changed(self):
        self.image_bfp = self._test_images.get(self.test_image_name)

    overlay_bfp_warp_arrows = Instance(OverlayTraitWarpArrows)
    overlays_bfp = List(Instance(OverlayTrait), transient=True)

    def _overlay_bfp_warp_arrows_default(self):
        return OverlayTraitWarpArrows()

    def _overlays_bfp_default(self):
        return [self.overlay_bfp_warp_arrows,
                ]

    overlay_warped_grid = Instance(OverlayTraitGrid)
    overlays_warped = List(Instance(OverlayTrait), transient=True)
    def _overlay_warped_grid_default(self):
        return OverlayTraitGrid()

    def _overlays_warped_default(self):
        return [self.overlay_warped_grid,
                ]



    calibrator = Instance(CalibratorBFPGrid, transient=True)

    def _calibrator_default(self):
        print("init calibrator")
        cal = CalibratorBFPGrid(parent=self)
        cal.load_settings()
        return cal

    def close(self, info, is_OK):
        print("closing down", info.ui.id)
        self.dump_settings()
        self.calibrator.dump_settings()
        return True

    view = View(
        HSplit(
            VGroup(
                Item('image_bfp',
                     editor=GLImageWithOverlaysEditor(overlays='overlays_bfp'),
                     show_label=False, springy=True, width=0.8),
                Item('image_bfp_warped',
                     editor=GLImageWithOverlaysEditor(overlays='overlays_warped'),
                     show_label=False, springy=True, width=0.8),
                ),
            VSplit(
                Item('test_image_name',
                     editor = EnumEditor(name='object._test_images_names'),
                     ),
                Item('calibrator',
                     style='custom', show_label=False),
                Item('overlays_bfp',
                     style='custom',
                     editor=ListEditor(use_notebook=True),
                     springy=True, show_label=False,
                            ),
                Item('overlays_warped',
                     style='custom',
                     editor=ListEditor(use_notebook=True),
                     springy=True, show_label=False,
                            ),
                ),
            ),
        resizable=True,
        )
    
def test_controller():
    c = TestController()
    c.load_settings()
    c.edit_traits()
    #c.configure_traits()
    return c

def start_event_loop():
    from IPython.lib.guisupport import start_event_loop_qt4, is_event_loop_running_qt4
    from pyface.qt import QtCore
    if not is_event_loop_running_qt4():
        print('qt4 event loop not running')
        print('starting event loop via IPython.lib.guisupport.start_event_loop_qt4')
        start_event_loop_qt4(QtCore.QCoreApplication.instance())
    else:
        print('running qt4 event loop detected')


# main program
if __name__ == '__main__':



    print("type in %pylab   !!!!!!")
    c = test_controller()
    cal = c.calibrator

    c.test_image_name='dots'
    c.calibrator.calculate=True

    c.overlay_warped_grid.center=(511.5, 511.5)
    c.overlay_warped_grid.extend=1023
    start_event_loop()

    #plt.imshow(cal.dbgimg1, cmap=plt.cm.gray)
    #plt.plot(cal.dbgdata1[:,0],
    #         cal.dbgdata1[:,1], 'r+')
    #plt.show()


    
