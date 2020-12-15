from __future__ import print_function
import numpy as np
from traits.api import (HasTraits, Int, Float, Str, Range, Python,
                        Bool, Event, Button,
                        Enum, Array, List, Dict, Tuple,
                        DelegatesTo, Instance, Property,
                        property_depends_on
                        )

import pyopencl as cl
import pyopencl.array as cla

cl_platform = cl.get_platforms()[0]
cl_device = cl_platform.get_devices(cl.device_type.GPU)[0]
cl_context = cl.Context([cl_device])
cl_queue = cl.CommandQueue(cl_context)

class GPUWarp(HasTraits):
    #settings_id = 'gpuwarp'
    shape = (512, 512) #(1024, 1024) #
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


# main program
if __name__ == '__main__':
    pass

    
