import attr
import numpy as np
from cltools import CLConsumer
from multiareapropagator import MultiAreaField, MultiAreaFocalToFarfieldPropagatorGPU
from fringer import FringerGPU
from propagator import FresnelPropagatorGPU, Field, ParaxialOptics

from transmission import TransmissionFresnelGPU
from convolverGPU import ConvolverGPU
import ABCD
import warnings

import pyopencl.array as cla

def double_gaussian_kernel(N = 1024, sigma_1 = 1., sigma_2 = 10., p2 = 0.2):
    # sum of two 2d gaussians, normalized
    # --> camera contrast model
    k = np.fft.fftfreq(N)  # 0 ... 0.5 -0.5 ... 0
    Kx, Ky = np.meshgrid(k, k, sparse=True)
    sigma_1 = sigma_1/N
    sigma_2 = sigma_2/N
    g1 = np.exp(-(Kx*Kx)/(2*sigma_1**2)) * np.exp(-(Ky*Ky)/(2*sigma_1**2))
    g2 = np.exp(-(Kx*Kx)/(2*sigma_2**2)) * np.exp(-(Ky*Ky)/(2*sigma_2**2))

    kernel = (1-p2)*g1*(1./g1.sum()) + p2*g2*(1./g2.sum())
    return kernel


def transposer(x):
    "return transposed copy of input array"
    return x.T.copy()


@attr.s
class FieldRetrieverGPU(CLConsumer):
    R0 = attr.ib(kw_only = True) # Radius corresponding to maximal angle where light is transmitted in the back focal plane
    
    slm_size = attr.ib(kw_only=True,
                       default=(15e-6 * 512,) * 2)
    slm_phase = attr.ib(kw_only=True, converter=transposer)  #: array : phase displayed on SLM
    slm_phase0 = attr.ib(kw_only=True, converter=transposer)

    slm_oversample = attr.ib(kw_only=True,
                             default=8)
    slm_field = attr.ib(init=False)
    @slm_field.default
    def _slm_field_default(self):
        return Field(shape=(512*self.slm_oversample, 512*self.slm_oversample),
                     size=self.slm_size)

    I0 = attr.ib(kw_only=True)  #: array : observed intensity empty system

    fringer = attr.ib(kw_only=True)

    @fringer.default
    def _fringer_default(self):
        return FringerGPU(cl_context=self.cl_context,
                          oversample=self.slm_oversample)

    fourier_plane_shape = attr.ib(kw_only=True, default=(1024, 1024))
    fourier_plane_size = attr.ib(kw_only=True, default=(10e-3,) * 2)

    #: Field : light field at first intermediate focal (Fourier) plane, where stops (iris, zero order block) are placed
    fourier_field: Field = attr.ib(kw_only=True)

    @fourier_field.default
    def _fourier_field_default(self):
        return Field(shape=self.fourier_plane_shape, size=self.fourier_plane_size)

    fourier_plane_mask = attr.ib(kw_only=True)

    #: Field : light field at pupil of objective lens
    pupil_field_shape = attr.ib(kw_only=True, default=(1024, 1024))
    pupil_field: Field = attr.ib(kw_only=True)

    @pupil_field.default
    def _pupil_field_default(self):
        return Field(shape=self.pupil_field_shape, size=self.slm_size)

    objective_focal_length = attr.ib(default=180e-3 / 60, kw_only=True)  #: float : objective lens focal length

    propagator_pupil_to_object = attr.ib(init=False,
                                         kw_only=True)  #: FresnelPropagatorGPU : propagator from pupil objective lens to object plane

    object_plane_shape = attr.ib(kw_only=True, default=(1024, 1024))
    object_plane_size = attr.ib(init=False, kw_only=True)
    @object_plane_size.default
    def _object_plane_size_default(self):
        f_relay = 300e-3  
        f_obj = self.objective_focal_length
        M = f_relay/f_obj  # demagnification objective
        return tuple( (s/M for s in self.fourier_plane_size) )
    object_plane_axial_position = attr.ib(kw_only=True, default=0.)  #: float : axial position of object plane (active traps)

    #: MultiAreaField : field at focal plane of objective lens, with subfields
    object_multiareafield: MultiAreaField = attr.ib(kw_only=True)

    @object_multiareafield.default
    def _object_multiareafield_default(self):
        return MultiAreaField(shape=self.object_plane_shape, size=self.object_plane_size)

    object_subfields_0 = attr.ib(init=False, kw_only=True)  #: list[Field] : copy of initial values subfields
    object_subfields_combined_0 = attr.ib(init=False, kw_only=True)  #: array : initial value combined subfields

    #: MultiAreaFocalToFarfieldPropagatorGPU : propagator (for patches and full field) from object plane to farfield
    propagator_object_to_farfield = attr.ib(init=False, kw_only=True)

    far_field = attr.ib(kw_only=True)  #: Field : farfield

    @far_field.default
    def _far_field_default(self):
        return Field(shape=self.I0.shape, size=self.slm_size)

    far_field_ref = attr.ib(init=False,
                            kw_only=True)  #: Field : reference farfield (propagated without subfields zeroed out)
  
  
    mask_bfp = attr.ib(kw_only = True)
    cl_mask_bfp = attr.ib(kw_only = True)
    @mask_bfp.default
    def _mask_bfp_default(self):
        x, y = np.linspace(-1,1,self.far_field.shape[0]), np.linspace(-1,1,self.far_field.shape[1])
        X,Y = np.meshgrid(x,y, sparse=True, indexing='xy')
        return np.where((X**2+Y**2)<self.R0**2,1,0)
    
    @cl_mask_bfp.default
    def _cl_mask_bfp_default(self):
        return cla.to_device(ary=self.mask_bfp.astype(np.float32),
                             queue=self.cl_queue, allocator=self.cl_allocator)
    

    detector_psf = attr.ib(kw_only=True)
    

    def _init_transmission_detection(self):
        self.transmission_detection = TransmissionFresnelGPU(shape = self.far_field.shape,
                                                r0 = self.R0,
                                                cl_context = self.cl_context,
                                                cl_queue = self.cl_queue,
                                                cl_profiler = self.cl_profiler,
                                                cl_allocator = self.cl_allocator)

    def _init_propagators(self):
        f = 300e-3  # focal length relay optics SLM to pupil  
        dz = self.object_plane_axial_position

        optics_relay = ParaxialOptics(ABCD=ABCD.lens_at_f(f))
        self.propagator_slm_to_fourier = FresnelPropagatorGPU(self.slm_field, self.fourier_field,
                                                              paraxial_optics=optics_relay,
                                                              cl_queue=self.cl_queue)
        self.propagator_fourier_to_pupil = FresnelPropagatorGPU(self.fourier_field, self.pupil_field,
                                                                paraxial_optics=optics_relay, cl_queue=self.cl_queue,
                                                                )

        f_obj = self.objective_focal_length
        optics_objective = ParaxialOptics(
            ABCD=ABCD.to_inf(d1=f_obj, f=f_obj, d2=f_obj+dz),
            )
        self.propagator_pupil_to_object = FresnelPropagatorGPU(self.pupil_field, self.object_multiareafield.field,
                                                               paraxial_optics=optics_objective,
                                                               cl_queue=self.cl_queue,
                                                               )


    def _init_fields(self):
        """
        initialize fields

        1) fringe slm field
        2) propagate to Fourier
        3) apply mask (iris, zero order block)
        4) propagate to pupil objective lens
        5) propagate to object plane


        Returns
        -------

        """
        def upsample(a, os):
            s0, s1 = a.shape
            au = np.empty(shape=(s0 * os, s1 * os), dtype=np.float32)
            for m in range(os):
                for n in range(os):
                    au[m::os, n::os] = a
            return au
        offset_fringer = 0.3  

        # calculate fringed SLM phase
        slm_phase_fringed = self.fringer.fringe2d(np.nan_to_num(self.slm_phase.T) + offset_fringer).T.copy() - offset_fringer



        # Field at SLM: from intensity empty (compensated for condenser transmission) and fringed phase
        # upsampled to same size
        I0_slm = np.nan_to_num(self.I0 / self.transmission_detection.transmission,
                                   nan=0., neginf=0., posinf=0.)
        
        A0 = upsample(np.sqrt(I0_slm),
                           os=slm_phase_fringed.shape[0] // I0_slm.shape[0])
        
        slm_phase0 = upsample(np.nan_to_num(self.slm_phase0, copy=False), os=slm_phase_fringed.shape[0] //self.slm_phase0.shape[0]).T 
        E_slm = A0 * np.exp(2j * np.pi * (slm_phase_fringed - slm_phase0)) 
        self.slm_field.field = E_slm

        # propagate SLM to Fourier plane (iris, zero order block)
        self.propagator_slm_to_fourier.propagate()
        self.fourier_field.field *= self.fourier_plane_mask
        self.propagator_fourier_to_pupil.propagate()
        self.propagator_pupil_to_object.propagate()


    def _init_patches(self, pos, patch_size=5e-6):
        self.patch_size = patch_size
    
        pos_xy = pos[:,:2]
        if np.isscalar(patch_size):
            patch_size = np.ones_like(pos_xy)*patch_size

        f = self.object_multiareafield.field

        idxA = f.find_idx_from_pos(pos_xy - .5 * patch_size)
        idxB = f.find_idx_from_pos(pos_xy + .5 * patch_size)
        self.object_multiareafield.subfields = []
        self.object_multiareafield.subfields_masks = []
        i = 0
        for (sxm, sym), (sxp, syp) in zip(idxA, idxB):
            subfield = f[sxm:sxp + 1, sym:syp + 1]  
            subfieldmask = (np.square((subfield.X[0] - subfield.center[0])) +
                            np.square((subfield.X[1] - subfield.center[1]))) < (
                                       patch_size[i,0] / 2) ** 2  
            self.object_multiareafield.append_subfield(subfield, mask=subfieldmask)
            i += 1

        self.object_multiareafield.combine_subfields()

        # store initial values of patches
        self.object_subfields_combined_0 = self.object_multiareafield.subfields_combined.copy()

        self.object_subfields_0 = [f.field.copy() * self.object_multiareafield.subfields_masks[i] for i, f in
                                   enumerate(self.object_multiareafield.subfields)]

        # check if patches overlap
        ind_mask_sum = 0
        for ind_mask in self.object_multiareafield.subfields_masks:
            ind_mask_sum += ind_mask.sum()

        tot_mask_sum = self.object_multiareafield.mask_subfields_combined.sum()

        if tot_mask_sum < ind_mask_sum:
            warnings.warn('Patches overlap, careful when propagating individual patches', Warning)


    def _init_propagator_farfield(self):
        self.propagator_object_to_farfield = MultiAreaFocalToFarfieldPropagatorGPU(cl_context=self.cl_context,
                                                                                   cl_queue=self.cl_queue,
                                                                                   focal=self.object_multiareafield,
                                                                                   pupil=self.far_field,
                                                                                   optics=self.propagator_pupil_to_object.optics,
                                                                                   )

    def _init_cl_arrays(self):
        self.cl_farfield_intensity = cla.empty(self.cl_queue,
                                               shape=self.far_field.shape,
                                               dtype=np.float32,
                                               allocator=self.cl_allocator,
                                               )

    def _init_convolver_detector(self):
        cl_detector_psf = cla.to_device(self.cl_queue, self.detector_psf, allocator=self.cl_allocator)
        self.convolver_detector = ConvolverGPU(cl_context = self.cl_context,
                                               cl_queue = self.cl_queue,
                                               cl_profiler = self.cl_profiler,
                                               cl_allocator = self.cl_allocator,
                                               x_cl=self.cl_farfield_intensity,
                                               k_cl=cl_detector_psf)
    

    def _init_reference_field(self, scale_ref=1):
        # clear object patches
        for subfield, mask in zip(self.object_multiareafield.subfields, self.object_multiareafield.subfields_masks):
            np.copyto(subfield.field, 0, where=mask)

        # obtain reference field
        self.propagator_object_to_farfield.propagator_full_field.propagate()

        self.object_field_ref = self.object_multiareafield.field.copy()
        self.far_field.field *= scale_ref
        self.far_field_ref = self.far_field.copy()

        self.cl_far_field_ref = cla.to_device(self.cl_queue, self.far_field_ref.field, allocator=self.cl_allocator)

    def init_all(self, pos, patch_size=5e-6, init_reference=True):
        self._init_transmission_detection()
        self._init_propagators()
        self._init_fields()
        self._init_patches(pos, patch_size)
        self._init_propagator_farfield()
        self._init_cl_arrays()
        self._init_convolver_detector()
        if init_reference:
            self._init_reference_field()

    def copyto_object_subfields_combined(self, subfields: list) -> np.ndarray:
        # write subfields into patches, return combined subfields (zero outside), has side effects on self.focal.subfields_combined
        self.object_multiareafield.subfields_combined.field.fill(0) 
        for subfield, data, mask in zip(self.object_multiareafield.subfields, subfields,
                                        self.object_multiareafield.subfields_masks):
            np.copyto(subfield.field[:], data, where=mask)

        return self.object_multiareafield.subfields_combined.field

    def calc_gradient_gpu(self, subfields_combined, cl_I_cam_measured=None, forward_only=False, individual_farfields=False):
        """
        calculate objective (forward model) and gradient (gradient backpropagation) with respect to subfields_combined


        Parameters
        ----------
        individual_farfields
        subfields_combined
        cl_I_cam_measured
        forward_only

        Returns
        -------
        (objective, gradient : np.array)
        (objective, I_cam : pyopencl.array.Array) if forward_only=True, and saves far_field
        (E, I) if individual_farfields==True

        """

        # propagate to far-field
        self.propagator_object_to_farfield.propagator_combined.cl_field1.set(subfields_combined, self.cl_queue)
        self.propagator_object_to_farfield.propagator_combined.propagate_gpudata()

        cl_far_field = self.propagator_object_to_farfield.propagator_combined.cl_field2

        # add reference field in new array
        if individual_farfields:
            cl_E = cl_far_field

        else:
            cl_E = cl_far_field + self.cl_far_field_ref


        cl_I = abs(cl_E)
        cl_I *= cl_I

        # apply detector model
        cl_I_cam = self.convolver_detector.convolve_gpu(cl_I)
            
        if individual_farfields: #returns intensity without transmission applied
            return cl_far_field.get(), (cl_I_cam*self.cl_mask_bfp).get()

        # apply condenser transmission
        self.transmission_detection.apply_inline_gpu(cl_I_cam) 

        cl_residuum = cl_I_cam - cl_I_cam_measured
        objective = cla.dot(cl_residuum, cl_residuum, queue=self.cl_queue).get()

        if forward_only: # return objective, I_cam with and withou tranmission applied
            self.far_field.field[:] = cl_E.get()  #store far_field
            cl_I_cam_no_transmission = self.convolver_detector.convolve_gpu(abs(cl_E)**2)*self.cl_mask_bfp
            return objective, cl_I_cam, cl_I_cam_no_transmission

        # backward gradient propagation
        self.transmission_detection.apply_inline_gpu(cl_residuum)
        
        cl_I_cam_bar = self.convolver_detector.convolve_bar_gpu(cl_residuum)
        
        cl_Ebar = 2*cl_E*cl_I_cam_bar
        self.propagator_object_to_farfield.propagator_combined.gradient_backprop_gpudata(cl_Ebar)

        subfields_combined_gradient = self.propagator_object_to_farfield.propagator_combined.cl_field1.get()


        subfields_combined_gradient *= self.object_multiareafield.mask_subfields_combined

        return objective, subfields_combined_gradient


    def retrieve_field(self, I_cam_measured, object_subfields_0=None, iterations=100, stepsize=100, momentum=0.8):
        '''
        perform phase retrieval

        I_bfp : float
            measured BFP intensity
        iterations, stepsize, momentum : int, float, float, optional
            settings for optimization algorithm
        '''
        self.iterations = iterations
        self.stepsize = stepsize
        self.momentum = momentum

        # init subfields with field for empty trap
        if object_subfields_0 is None:
            object_subfields_0 = self.object_subfields_0
        subfields_combined = self.copyto_object_subfields_combined(object_subfields_0)

        cl_I_cam_measured = cla.to_device(self.cl_queue, ary=I_cam_measured, allocator=self.cl_allocator)

        objectives = []
        v = np.zeros_like(subfields_combined)
        for k in range(iterations):
            # GPU
            objective, gradient = self.calc_gradient_gpu(subfields_combined + momentum * v, cl_I_cam_measured)
            objectives.append(objective)
            v = momentum * v - stepsize * gradient
            subfields_combined += v
            self.progress_iter = k / iterations

        objective_final, cl_I_cam_fit, cl_I_cam_fit_no_tranmission = self.calc_gradient_gpu(np.ascontiguousarray(subfields_combined),
                                                               cl_I_cam_measured, forward_only=True)

        self.object_multiareafield.subfields_combined.field[:] = subfields_combined  

        objectives.append(objective_final)

        self.log = np.array(objectives)

        return cl_I_cam_fit.get(), cl_I_cam_fit_no_tranmission.get()


    def calculate_individual_farfields(self, just_intesities = True):
        #calculates individual intensities at detector, and farfields (without detector and condenser transmission model)

        self.individual_farfield_intensities = []
        self.individual_farfields = []
        subfield_combined = self.object_multiareafield.subfields_combined
        for mask in self.object_multiareafield.masks_subfields_combined:
            E, I = self.calc_gradient_gpu(subfield_combined.field * mask, individual_farfields=True)
            self.individual_farfield_intensities.append(I.copy())
            if not just_intesities:
                self.individual_farfields.append(E.copy())


if __name__ == '__main__':
    pass
