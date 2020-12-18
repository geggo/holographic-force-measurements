#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
import mkl_fft

from .next_regular import next_regular_mul8
from .field import Field

import attr
import functools

import pyopencl as cl
import pyopencl.array as cla
import gpyfft
import numexpr as ne


@attr.s
class BasicOptics(object):
    wavelength = attr.ib(default=1e-6)


@attr.s
class ParaxialOptics(BasicOptics):
    ABCD = attr.ib(default=[[1, 0], [0, 1]], converter=np.asmatrix)



@attr.s
class HighNAOptics(BasicOptics):
    NA = attr.ib(default=1.2)
    refractive_index = attr.ib(default=1.33)

    @property
    def wavelength_medium(self):
        return self.wavelength / self.refractive_index

    @property
    def sin_theta_max(self):
        return self.NA / self.refractive_index


# TODO: make attr.s class
class FresnelPropagator(object):
    def __init__(self, field1, field2, paraxial_optics=None):
        self.optics = paraxial_optics

        assert field1.dtype == field2.dtype, 'fields need to have same precision'
        assert field1.ndim == field2.ndim, 'fields need to have same rank'

        self.field1 = field1
        self.field2 = field2
        self.dtype = field1.dtype
        self.ndim = field1.ndim
        self.complexdtype = field1.complexdtype

        self._init()

    def _init(self):
        self._init_data()
        self._init_work_arrays()
        self._init_fft()

    def _init_data(self):

        # array sizes
        N1 = np.array(self.field1.shape)
        N2 = np.array(self.field2.shape)
        N12 = N1 + N2 - 1
        N12_pad = np.array([next_regular_mul8(int(n12)) for n12 in N12])

        # centered coordinates, unit spacing
        n1, n2, n12 = ([np.arange(n) - (n - 1) * 0.5 for n in N] for N in (N1, N2, N12))
        
        # shift and pad
        n12_pad = []
        for k in range(self.ndim):
            n_pad = np.zeros(N12_pad[k], self.dtype)  # aligned?
            n_pad[:N12[k]] = n12[k]
            n_pad = np.roll(n_pad, N2[k] + (N12_pad[k] - N12[k]))
            n12_pad.append(n_pad)

        self.N1, self.N2, self.N12, self.N12_pad = N1, N2, N12, N12_pad
        self.n1, self.n2, self.n12, self.n12_pad = n1, n2, n12, n12_pad

        # phase factors, convolution kernel, scalings
        delta1, delta2 = self.field1.delta, self.field2.delta
        (A, B), (C, D) = self.optics.ABCD.tolist()
        b = 1j * np.pi / (B * self.optics.wavelength)

        center1 = self.field1.center
        center2 = self.field2.center

        self.w1 = [np.exp(b * (A * d1 * d1 + d1 * d2) * n1 ** 2 + 2*b*n1*d1*c2) for (d1, d2, n1, c2) in zip(delta1, delta2, n1, center2)]
        self.w2 = [np.exp(b * (D * d2 * d2 + d1 * d2) * n2 ** 2 + 2*b*n2*d2*c1) for (d1, d2, n2, c1) in zip(delta1, delta2, n2, center1)]
        self.w12_pad = [np.exp(-b * d1 * d2 * n12_pad ** 2) for (d1, d2, n12_pad) in zip(delta1, delta2, n12_pad)]
        self.w12F_pad = [np.fft.fft(w) for w in self.w12_pad]

        self.W1 = np.meshgrid(*self.w1, sparse=True, copy=False, indexing='ij')
        self.W2 = np.meshgrid(*self.w2, sparse=True, copy=False, indexing='ij')
        self.W12F_pad = np.meshgrid(*self.w12F_pad, sparse=True, copy=False, indexing='ij')

        self.Sf = np.prod([np.sqrt(1. / (self.optics.wavelength * B)) * d1 for d1 in self.field1.delta])
        self.Sb = np.prod([np.sqrt(1. / (self.optics.wavelength * B)) * d2 for d2 in self.field2.delta])

    def _init_work_arrays(self):
        # work arrays
        #self.work = pyfftw.empty_aligned(self.N12_pad, dtype=self.complexdtype)
        #self.work_F = pyfftw.empty_aligned(self.N12_pad, dtype=self.complexdtype)
        self.work = np.empty(self.N12_pad, dtype=self.complexdtype)

        def make_view(N):
            sel = tuple(slice(None, n) for n in N)
            return self.work[sel]

        self.work_view1 = make_view(self.N1)
        self.work_view2 = make_view(self.N2)

    def _init_fft(self):
        pass

    def propagate(self):
        self.work.fill(0)
        self.work_view1[:] = self.field1.field

        for W in self.W1:
            self.work_view1 *= W

        self.work_F = mkl_fft.fftn(self.work, overwrite_x=True)

        for WF in self.W12F_pad:
            self.work_F *= WF

        self.work[:] = mkl_fft.ifftn(self.work, overwrite_x=True)

        for W in self.W2:
            self.work_view2 *= W

        self.field2.field = self.work_view2 * self.Sf

    def propagate_backward(self):
        self.work.fill(0)
        self.work_view2[:] = self.field2.field

        for W in self.W2:
            self.work_view2 *= W.conj()

        self.work_F = mkl_fft.fftn(self.work, axes=range(-self.ndim, 0), overwrite_x=True)

        for WF in self.W12F_pad:
            self.work_F *= WF.conj()

        self.work[:] = mkl_fft.ifftn(self.work, axes=range(-self.ndim, 0), overwrite_x=True)

        for W in self.W1:
            self.work_view1 *= W.conj()

        self.field1.field = self.work_view1 * self.Sb


# TODO: see bnsdrive.py for better profiling
#Franziska
def profile(event, hint=''):
    try:
        event.wait()
        dt = 1e-6 * (event.profile.end - event.profile.start)
        #print('profiling %16s %6.2f'%(hint, dt), 'ms')
    except cl.RuntimeError as e:
        #print(e)
        pass

# TODO: derive from CLConsumer
class FresnelPropagatorGPU(FresnelPropagator):
    def __init__(self, field1, field2, paraxial_optics=None,
                 cl_context=None, cl_queue=None,
                 cl_profile=False):
        self.context = cl_context
        self.queue = cl_queue
        self.cl_profile = cl_profile
        super().__init__(field1, field2, paraxial_optics)

    def _init(self):
        # override parent method for custom initialization
        self._init_data()
        self._init_cl()
        self._init_cl_arrays()
        self._init_cl_fft()
        self._init_cl_kernels()

    def _init_cl(self):

        if self.queue is not None:  #queue given, take context from queue
            self.context = self.queue.context

        if self.context is None: #neither queue nor context given, create one
            platform = cl.get_platforms()[0]
            self.device = platform.get_devices(cl.device_type.GPU)[0]
            self.context = cl.Context([self.device])

        if self.queue is None:
            props = cl.command_queue_properties.PROFILING_ENABLE if self.cl_profile else 0
            self.queue = cl.CommandQueue(self.context, properties=props)
        else:
            if self.queue.properties & cl.command_queue_properties.PROFILING_ENABLE:
                # queue profiling enabled
                if not self.cl_profile:
                    print('disabling profiling')
                    self.cl_profile = False
            else:
                # queue profiling disabled
                if self.cl_profile:
                    print('cannot enable profiling because queue not configured for profiling, disabling it')
                    self.cl_profile = False

    def _init_cl_arrays(self):
        self.cl_w1 = [cla.to_device(self.queue, w.astype(self.complexdtype)) for w in self.w1]
        self.cl_w12F_pad = [cla.to_device(self.queue, w.astype(self.complexdtype)) for w in self.w12F_pad]
        self.cl_w2 = [cla.to_device(self.queue, w.astype(self.complexdtype)) for w in self.w2]

        self.cl_w1_conj = [cla.to_device(self.queue, w.astype(self.complexdtype).conj()) for w in self.w1]
        self.cl_w12F_pad_conj = [cla.to_device(self.queue, w.astype(self.complexdtype).conj()) for w in self.w12F_pad]
        self.cl_w2_conj = [cla.to_device(self.queue, w.astype(self.complexdtype).conj()) for w in self.w2]

        self.cl_work = cla.zeros(self.queue, tuple(self.N12_pad), self.complexdtype)
        self.cl_workF = cla.zeros_like(self.cl_work)

        self.cl_field1 = cla.empty(self.queue, tuple(self.N1), self.complexdtype)
        self.cl_field2 = cla.empty(self.queue, tuple(self.N2), self.complexdtype)

    def _init_cl_fft(self):
        self.cl_fftplan = gpyfft.fft.FFT(self.context, self.queue, self.cl_work, self.cl_workF)
        self.cl_ifftplan = gpyfft.fft.FFT(self.context, self.queue, self.cl_workF, self.cl_work)

    def _init_cl_kernels(self):

        source = open('propagator.cl', 'r').read()
        program = cl.Program(self.context, source)
        program.build()
        build_info = program.get_build_info(self.context.devices[0],
                                            cl.program_build_info.LOG)
        if build_info:
            print("warning: non-empty build info:")
            print(build_info)

        self.cl_kernel_premul = program.premul_field_2d
        #self.cl_kernel_premul.set_scalar_arg_dtypes(('uint32',) * 2 + (None,) * 4)

        self.cl_kernel_mul = program.multiply_workF_2d

        self.cl_kernel_postmul = program.postmul_field_2d
        #self.cl_kernel_postmul.set_scalar_arg_dtypes(('uint32',) * 2 + (None,) * 2 + ('float32',) + (None,) * 2)


    def propagate(self):
        # copy field data host->gpu
        evt0 = cl.enqueue_copy(self.queue, self.cl_field1.data, np.ascontiguousarray(self.field1.field))
        profile(evt0, 'copy to device')

        self.propagate_gpudata()

        # copy from gpu to numpy array
        self.field2.field = self.cl_field2.get()

    def propagate_gpudata(self):
        gws12p = (int(self.N12_pad[1]),  # global work size, reverse order!!
                  int(self.N12_pad[0]))

        local_work_size = (8, 8)

        # prepare work array: fill in zero padded copy of field1, premultiply with w1
        self.cl_kernel_premul.set_args(
            np.uint32(self.N1[0]), np.uint32(self.N1[1]),
            self.cl_w1[0].data,
            self.cl_w1[1].data,
            self.cl_field1.data,
            self.cl_work.data,
        )

        evt1 = cl.enqueue_nd_range_kernel(
            self.queue,
            self.cl_kernel_premul,
            global_work_size=gws12p,
            local_work_size=local_work_size
        )

        profile(evt1, 'premul')

        # forward fft of work array
        evt2, = self.cl_fftplan.enqueue()

        profile(evt2, 'fft')

        # multiply with kernel
        self.cl_kernel_mul.set_args(
            self.cl_w12F_pad[0].data, self.cl_w12F_pad[1].data,
            self.cl_workF.data
        )

        evt3 = cl.enqueue_nd_range_kernel(
            self.queue,
            self.cl_kernel_mul,
            global_work_size=gws12p,
            local_work_size=local_work_size
        )

        profile(evt3, 'mul')

        ## dbg_workF = self.cl_workF.get()

        # backward fft
        evt4, = self.cl_ifftplan.enqueue(  #
            forward=False
        )

        profile(evt4, 'ifft')

        ## np.testing.assert_almost_equal(np.fft.ifft2(dbg_workF), self.cl_work.get(), 5)

        # extract field2, multiplied with w2
        self.cl_kernel_postmul.set_args(
            np.uint32(self.N12_pad[0]), np.uint32(self.N12_pad[1]),
            self.cl_w2[0].data, self.cl_w2[1].data,
            np.float32(self.Sf),
            self.cl_work.data,
            self.cl_field2.data
        )
        evt5 = cl.enqueue_nd_range_kernel(self.queue,
                                          self.cl_kernel_postmul,
                                          global_work_size=(int(self.N2[1]),  # reverse order!!!
                                                            int(self.N2[0])),
                                          local_work_size=local_work_size)
        profile(evt5, 'postmul')


    def gradient_backprop(self, field2_grad : np.ndarray):
        # copy field2 gradient data host->device
        evt0 = cl.enqueue_copy(self.queue, self.cl_field2.data, field2_grad)  # TODO: replace by self.cl_field2.set_async

        self.gradient_backprop_gpudata(self.cl_field2)

        field1grad = self.cl_field1.get()
        return field1grad


    def gradient_backprop_gpudata(self, cl_field2_grad : cla.Array):
        # assume: called (immediately after propagate)
        # ident zu propagate, aber alles conjugiert, und field1/field2 vertauscht !?

        gws12p = (int(self.N12_pad[1]),  # reverse order!!
                  int(self.N12_pad[0]))

        local_work_size = (8, 8)

        # prepare work array: fill in padded copy of field2 gradient, premultiply with w2
        self.cl_kernel_premul.set_args(
            np.uint32(self.N2[0]), np.uint32(self.N2[1]),
            self.cl_w2_conj[0].data,
            self.cl_w2_conj[1].data,
            cl_field2_grad.data,
            self.cl_work.data,
        )

        evt1 = cl.enqueue_nd_range_kernel(
            self.queue,
            self.cl_kernel_premul,
            global_work_size=gws12p,
            local_work_size=local_work_size
        )

        # profile(evt1, 'premul')

        # forward fft of work array
        evt2, = self.cl_fftplan.enqueue(
            #forward=True,
        )

        # profile(evt2, 'fft')

        # multiply with kernel
        self.cl_kernel_mul.set_args(
            self.cl_w12F_pad_conj[0].data, self.cl_w12F_pad_conj[1].data,
            self.cl_workF.data
        )

        evt3 = cl.enqueue_nd_range_kernel(
            self.queue,
            self.cl_kernel_mul,
            global_work_size=gws12p,
            local_work_size=local_work_size
        )

        # profile(evt3, 'mul')

        ## dbg_workF = self.cl_workF.get()

        # backward fft
        evt4, = self.cl_ifftplan.enqueue(  #
            forward=False
        )

        ## np.testing.assert_almost_equal(np.fft.ifft2(dbg_workF), self.cl_work.get(), 5)

        # extract field1, multiplied with w1_conj
        self.cl_kernel_postmul.set_args(
            np.uint32(self.N12_pad[0]), np.uint32(self.N12_pad[1]),
            self.cl_w1_conj[0].data, self.cl_w1_conj[1].data,
            np.float32(self.Sf),
            # np.float32(self.Sb), # ???? proper scaling ?
            self.cl_work.data,
            self.cl_field1.data
        )
        evt5 = cl.enqueue_nd_range_kernel(self.queue,
                                          self.cl_kernel_postmul,
                                          global_work_size=(int(self.N1[1]),  # reverse order!!!
                                                            int(self.N1[0])),
                                          local_work_size=local_work_size)

        evt5.wait()

        # profile(evt5, 'postmul')



class FocusPropagator(object):
    """
    High-NA light field propagator from pupil to Focus.

    Boruah, B.R., and M.A.A. Neil.
    “Focal Field Computation of an Arbitrarily Polarized Beam Using Fast Fourier Transforms.”
    Optics Communications 282, no. 24 (December 2009): 4660–67.
    https://doi.org/10.1016/j.optcom.2009.09.019.

    """

    def __init__(self, pupil_field, focus_field,
                 optics=HighNAOptics()
                 ):
        self.optics = optics  # store

        field1 = pupil_field
        field2 = focus_field

        assert field1.dtype == field2.dtype, 'fields need to have same precision'
        assert field1.ndim == field2.ndim, 'fields need to have same rank'
        assert field1.vectorial == field2.vectorial, 'both input fields need to have same dimensionality (scalar or vectorial)'
        assert (field1.center == 0.).all(), 'pupil field needs to be centered around 0'

        self.field1 = field1
        self.field2 = field2
        self.dtype = field1.dtype
        self.ndim = field1.ndim
        self.vectorial = field1.vectorial
        self.complexdtype = field1.complexdtype

        self.dz = 0

        self._init()

    def _init(self):
        self._init_data()
        self.init_work_arrays()
        self._init_fft()

        if self.ndim > 1:
            self.init_highNA()


    def _init_data(self):

        # array shapes
        N1 = np.array(self.field1.shape)
        N2 = np.array(self.field2.shape)
        N12 = N1 + N2 - 1  # mininum size for work array: minimum size N1+N2-1
        N12_pad = np.array([next_regular_mul8(int(n12)) for n12 in N12])  # extended size for better FFT performance

        # -1,0,1 for N=3
        # -1.5, -0.5, 0.5, 1.5 for N=4
        # TODO: test: odd (ok?), even, n1/2 should be related to field1/2.x !?
        # also create entries for intermediate work array

        # centered coordinates, unit spacing
        n1, n2, n12 = ([np.arange(n) - (n - 1) * 0.5 for n in N] for N in (N1, N2, N12))

        n12_pad = []
        for k in range(self.ndim):
            n_pad = np.zeros(N12_pad[k], self.dtype)  # aligned?
            n_pad[:N12[k]] = n12[k]
            n_pad = np.roll(n_pad, N2[k] + (N12_pad[k] - N12[k]))
            # TODO: more efficient solution
            # Beobachtung: l_pad fängt immer mit -(N-M) an,
            # dann bis + (N+M), dann am Ende die fehlenden -(N+M) bis -(N-M)-1
            # ??wie schaut das für Daten mit gerader Anzahl Elemente aus?
            n12_pad.append(n_pad)

        self.N1, self.N2, self.N12, self.N12_pad = N1, N2, N12, N12_pad
        self.n1, self.n2, self.n12, self.n12_pad = n1, n2, n12, n12_pad

        # d = 2pi i * dx / lamda_medium * dk/k0
        # dx, dk: spacing grid in focus/pupil plane
        # k0: radius pupil sphere (corresponding to light at right angle)

        d12 = 2j * np.pi * self.field1.delta / self.optics.wavelength_medium * self.field2.delta / self.field1.r0
        c1, c2 = self.field1.center, self.field2.center

        #s = p1.r0 * P.optics.wavelength_medium
        #px, py = kx * s, ky * s

        dc = 2j*np.pi * self.field1.delta/(self.optics.wavelength_medium*self.field1.r0)

        # TODO center
        self.w1 = [np.exp(0.5 * d12 * n1 * n1  + dc*c2*n1) for d12, n1, c2, dc in zip(d12, n1, c2, dc)]  # phase factors
        self.w2 = [np.exp(0.5 * d12 * n2 * n2) for d12, n2 in zip(d12, n2)]  # phase factors
        self.w12_pad = [np.exp(-0.5 * d12 * n12_pad * n12_pad) for d12, n12_pad in zip(d12, n12_pad)]  # convolution kernels
        # TODO: set to zero where n12_pad is zero padded???
        # TODO: normalize, for energy conservation?

        # for broadcasting extend shape to (1,..,N,..,1)
        self.W1 = np.meshgrid(*self.w1, sparse=True, copy=False, indexing='ij')
        self.W2 = np.meshgrid(*self.w2, sparse=True, copy=False, indexing='ij')

        # Fourier transforms of convolution kernels
        Fw12_pad = [np.fft.fft(w) for k, w in enumerate(self.w12_pad)]  # TODO: normalization?
        self.FW12_pad = np.meshgrid(*Fw12_pad, sparse=True, copy=False, indexing='ij')  #

    def init_work_arrays(self):
        shape = tuple(self.N12_pad)
        if self.vectorial:
            shape = (3,) + shape
        self.work = np.empty(shape, dtype=self.complexdtype)

        def make_view(N):
            sel = tuple(slice(None, n) for n in N)
            if self.vectorial:
                sel = (slice(None),) + sel
            return self.work[sel]

        self.work_view1, self.work_view2 = make_view(self.N1), make_view(self.N2)

    def _init_fft(self):
        self.fft_forward = functools.partial(mkl_fft.fftn, axes=range(-self.ndim,0), overwrite_x=True)
        self.fft_backward = functools.partial(mkl_fft.ifftn, axes=range(-self.ndim,0), overwrite_x=True)
        #self.fft_forward = functools.partial(np.fft.fftn, axes=range(-self.ndim,0))
        #self.fft_backward = functools.partial(np.fft.ifftn, axes=range(-self.ndim,0))

    def init_highNA(self):
        assert self.ndim == 2
        k0 = self.field1.r0

        kx, ky = self.field1.X  # pupil coordinates

        kr2 = np.square(kx) + np.square(ky)

        mask = kr2 < k0 * k0  # mask (0) point outside pupil

        ikr2 = 1. / kr2 * mask  # problem at origin
        mask_fix = np.where(~np.isfinite(ikr2))  # problem points
        ikr2[mask_fix] = 0  # to suppress warnings about NaN operations

        kzn = np.sqrt((1 - 1. / (k0 * k0) * kr2) * mask)  # kz normalized, kz/k0

        # for linear polarized light along x
        Gxx = (np.square(kx) * kzn + np.square(ky)) * ikr2
        Gxy = kx * ky * (kzn - 1) * ikr2
        Gxz = 1. / k0 * kx * mask  # TODO: mask not strictly necessary (but gives compatible shape, for debug)

        # for linear polarized light along y
        Gyx = Gxy  # TODO Vorzeichen?
        Gyy = (np.square(kx) + np.square(ky) * kzn) * ikr2
        Gyz = 1. / k0 * ky * mask  # TODO: see above

        #Gzx = -Gxz
        #Gzy = -Gyz
        Gzz = kzn

        # fix invalid values at center / outside
        Gxx[mask_fix] = 1
        Gxy[mask_fix] = 0
        Gyy[mask_fix] = 1
        # Gyx[mask_fix] = 0 #fix already done done in Gxy (alias)
        Gzz[mask_fix] = 1  # ????

        # TODO: exp( 1j*kz*zp) missing (defocus)
        # TODO: sqrt(k0/kz) missing ???

        # apodization: sqrt(cos(theta)) *
        G0 = 1. / np.sqrt(kzn)  #### Neil, ok (?)
        G0[~mask] = 0
        self.G0 = G0

        G0_inv = np.sqrt(kzn)
        G0_inv[~mask] = 0
        self.G0_inv = G0_inv

        # TODO: make array
        self.G = [[Gxx,  None, None],
                  [Gxy,  Gyy,  None],
                  [Gxz,  Gyz,  Gzz]]

        self.kzn = kzn

    def set_defocus(self, dz):
        self.dz = dz
        self.W1_defocus = np.exp(2j*np.pi*self.kzn * dz *1./self.optics.wavelength_medium)

    def propagate(self):
        # 0) copy field into work array
        self.work[:] = 0
        self.work_view1[:] = self.field1.field

        # TODO add defocus (like in propagate_vectorial)

        # 1) multiply along each axis with corresponding entry of
        # phase factors w1 (outer product, separates!)

        for W in self.W1:
            self.work_view1 *= W
        # TODO: offset *= exp(-2j*pi*n1*offset)

        # print self.work

        # 2) transform
        work_F = self.fft_forward(self.work)

        # 3) multiply with transformed kernel
        for FW in self.FW12_pad:
            work_F *= FW

        # 4) backward transform
        self.work[:] = self.fft_backward(work_F)

        # normalization factor 1/N12_pad, TODO: put somewhere else (FW12_pad?)
        #self.work *= 1. / np.prod(self.N12_pad)

        self.work *= np.prod(self.field1.delta * np.sqrt(2. / self.optics.wavelength_medium))

        # 5) apply phase factors
        for W in self.W2:
            self.work_view2 *= W

        # TODO: phasefaktor für offset pupil

        self.field2.field = self.work_view2

    def propagate_back(self):
        # 0) copy field into work array
        self.work[:] = 0
        self.work_view2[:] = self.field2.field  # copy field data into work1 array

        # 5)
        for W in self.W2:
            self.work_view2 *= W.conj()

        # 4)
        self.work_F = self.fft_forward(self.work)

        # 3)
        for FW in self.FW12_pad:
            self.work_F *= FW.conj()

        # 2)
        self.work[:] = self.fft_backward(self.work_F)
        #self.work *= 1. / np.prod(self.N12_pad)
        self.work *= np.prod(self.field2.delta * np.sqrt(2. / self.optics.wavelength_medium))

        # 1)
        for W in self.W1:
            self.work_view1 *= W.conj()

        # 0)
        self.field1.field = self.work_view1  # note: setter for .field property!

    def propagate_pupil_to_focal_vectorial(self):

        ((Gxx, Gyx, Gzx),
         (Gxy, Gyy, Gzy),
         (Gxz, Gyz, Gzz)) = self.G

        Ex, Ey, Ez = self.field1.field

        # 0) copy field into work array
        self.work.fill(0)

        self.work_view1[0] = Gxx * Ex + Gxy * Ey - Gxz * Ez
        self.work_view1[1] = Gxy * Ex + Gyy * Ey - Gyz * Ez
        self.work_view1[2] = Gxz * Ex + Gyz * Ey + Gzz * Ez

        # apply apodization
        self.work_view1 *= self.G0

        # TODO: shortcut: linear, circular polarization for scalar field: field*Px, field*Py, 0
        # e.g. P = [1, 1j] for circular polarization
        # TODO: what about Ez (from Ex, Ey, and phase gradient ????), Ez very small??)

        # 1) multiply along each axis with corresponding entry of
        # phase factors w1 (outer product, separates!)

        if self.dz:
            self.work_view1 *= self.W1_defocus  # apply defocus

        for W in self.W1:
            self.work_view1 *= W

        # 2) transform
        self.work_F = self.fft_forward(self.work)

        # 3) multiply with transformed kernel
        for FW in self.FW12_pad:
            self.work_F *= FW

        # 4) backward transform
        self.work[:] = self.fft_backward(self.work_F)

        # 5) apply phase factors
        for W in self.W2:
            self.work_view2 *= W

        # normalization
        self.work *= np.prod(self.field1.delta / np.sqrt(self.field1.r0) / np.sqrt(self.optics.wavelength)) # * np.sqrt(2. / self.optics.wavelength_medium))

        self.field2.field = self.work_view2

    def gradient_backprop_focal_to_pupil_vectorial(self):

        self.work.fill(0)
        self.work_view2[:] = self.field2.field

        # 5)
        for W in self.W2:
            self.work_view2 *= W.conj()

        # 4)
        self.work_F = self.fft_forward(self.work)

        # 3)
        for FW in self.FW12_pad:
            self.work_F *= FW.conj()

        # 2)
        self.work[:] = self.fft_backward(self.work_F)

        # 1)
        for W in self.W1:
            self.work_view1 *= W.conj()

        # undo defocus
        if self.dz:
            self.work_view1 *= self.W1_defocus.conj()

        # normalization TODO field1 or field2? (field1 strict gradient, but field2.delta (like in inverse propagation) better scaling?)
        self.work *= np.prod(self.field1.delta / np.sqrt(self.field1.r0) / np.sqrt(self.optics.wavelength))

        self.work_view1 *= self.G0 #.conj()

        ((Gxx, Gyx, Gzx),
         (Gxy, Gyy, Gzy),
         (Gxz, Gyz, Gzz)) = self.G

        Ex, Ey, Ez = self.work_view1

        Ebar = np.empty_like(self.work_view1)
        Ebar[0] =  Gxx * Ex + Gxy * Ey + Gxz * Ez
        Ebar[1] =  Gxy * Ex + Gyy * Ey + Gyz * Ez
        Ebar[2] = -Gxz * Ex - Gyz * Ey + Gzz * Ez

        return Ebar

    def propagate_focal_to_pupil_vectorial(self):
        """field 2 to field 1"""

        self.work.fill(0)
        self.work_view2[:] = self.field2.field

        # 5)
        for W in self.W2:
            self.work_view2 *= W.conj()  # multiply mit conjugate instead of dividing, W is on unit circle

        # 4)
        self.work_F = self.fft_forward(self.work)

        # 3)
        for FW in self.FW12_pad:
            self.work_F *= FW.conj()

        # 2)
        self.work[:] = self.fft_backward(self.work_F)

        # 1)
        for W in self.W1:
            self.work_view1 *= W.conj()

        # undo defocus
        if self.dz:
            self.work_view1 *= self.W1_defocus.conj()

        # energy conserving normalization
        self.work *= np.prod(self.field2.delta / np.sqrt(self.field1.r0) / np.sqrt(self.optics.wavelength))

        self.work_view1 *= self.G0_inv

        ((Gxx, Gyx, Gzx),
         (Gxy, Gyy, Gzy),
         (Gxz, Gyz, Gzz)) = self.G

        Ex, Ey, Ez = self.work_view1

        field1_x, field1_y, field1_z = self.field1.field

        # transform
        #TODO: inverse of G is transpose?
        field1_x[:] =  Gxx * Ex + Gxy * Ey + Gxz * Ez
        field1_y[:] =  Gxy * Ex + Gyy * Ey + Gyz * Ez
        field1_z[:] = -Gxz * Ex - Gyz * Ey + Gzz * Ez


    def gradient_backpropagate_pupil_to_focal_vectorial(self):

        ((Gxx, Gyx, Gzx),
         (Gxy, Gyy, Gzy),
         (Gxz, Gyz, Gzz)) = self.G

        Ex, Ey, Ez = self.field1.field

        # 0) copy field into work array
        self.work.fill(0)

        self.work_view1[0] = Gxx * Ex + Gxy * Ey - Gxz * Ez
        self.work_view1[1] = Gxy * Ex + Gyy * Ey - Gyz * Ez
        self.work_view1[2] = Gxz * Ex + Gyz * Ey + Gzz * Ez

        # apply apodization
        self.work_view1 *= self.G0_inv

        # 1) multiply along each axis with corresponding entry of
        # phase factors w1 (outer product, separates!)

        if self.dz:
            self.work_view1 *= self.W1_defocus  # apply defocus

        for W in self.W1:
            self.work_view1 *= W

        # 2) transform
        self.work_F = self.fft_forward(self.work)

        # 3) multiply with transformed kernel
        for FW in self.FW12_pad:
            self.work_F *= FW

        # 4) backward transform
        self.work[:] = self.fft_backward(self.work_F)

        # 5) apply phase factors
        for W in self.W2:
            self.work_view2 *= W

        # normalization
        self.work *= np.prod(self.field2.delta / np.sqrt(self.field1.r0) / np.sqrt(self.optics.wavelength)) # * np.sqrt(2. / self.optics.wavelength_medium))

        Ebar = np.empty_like(self.work_view2)
        Ebar[:] = self.work_view2

        return Ebar


class RayleighSommerfeldPropagator(object):
    def __init__(self, field1, field2, optics, dz=1e-6):
        #threads = ne.detect_number_of_threads()
        #ne.set_num_threads(threads)
        assert field1.dtype == field2.dtype, 'fields need to have same precision'
        assert field1.ndim == field2.ndim, 'fields need to have same rank'
        self.optics = optics
        self.dz = dz
        self.field1 = field1
        self.field2 = field2
        self.dtype = field1.dtype
        self.ndim = field1.ndim
        self.complexdtype = field1.complexdtype

        self._init()

    def _init(self):
        self._init_data()
        self._init_work_arrays()
        # self._init_fft()

    def _init_data(self):

        # array sizes
        N1 = np.array(self.field1.shape)
        N2 = np.array(self.field2.shape)
        N12_pad = N1 + N2  # - 1
        # N12_pad = np.array([next_regular_mul8(n12) for n12 in N12])
        self.N1 = N1
        self.N2 = N2
        self.N12_pad = N12_pad

        # build kernel
        delta1, delta2 = self.field1.delta, self.field2.delta
        if (abs(delta1 - delta1).max() > 1e-7):
            print('sampling in source field is not equal to sampling in destination field')

        center1 = self.field1.center
        center2 = self.field2.center

        Lx_pad = (N12_pad[0] - 2) * delta1[0]
        Ly_pad = (N12_pad[1] - 2) * delta1[1]
        self.Lx_pad, self.Ly_pad = Lx_pad, Ly_pad

        # coordinates of padded array shifted by center
        x_pad = np.linspace(-Lx_pad / 2 - delta1[0] / 2, Lx_pad / 2 + delta1[0] / 2, N12_pad[0],
                            dtype=self.dtype) + np.float32(center2[0] - center1[0])
        y_pad = np.linspace(-Ly_pad / 2 - delta1[1] / 2, Ly_pad / 2 + delta1[1] / 2, N12_pad[1],
                            dtype=self.dtype) + np.float32(center2[1] - center1[1])
        self.x_pad, self.y_pad = x_pad, y_pad
        xx_pad, yy_pad = np.meshgrid(x_pad, y_pad, sparse=True, indexing='ij')
        g = np.zeros(N12_pad, dtype=self.complexdtype)

        if self.complexdtype == np.complex64:

            rr2 = ne.evaluate('xx_pad*xx_pad+yy_pad*yy_pad+dz*dz',
                              {'xx_pad': xx_pad, 'yy_pad': yy_pad, 'dz': np.float32(self.dz)})
            rr = ne.evaluate('sqrt(rr2)', {'rr2': rr2})
            ne.evaluate('dz/(two_s * pi_s * RR2) * (cos(k0 * RR) * one_s/RR + sin(k0*RR) * k0) * deltax * deltay',
                        {'RR': rr, 'RR2': rr2, 'k0': np.float32(2 * np.pi / self.optics.wavelength_medium),
                         'two_s': np.float32(2), 'one_s': np.float32(1), 'pi_s': np.float32(np.pi),
                         'dz': np.float32(self.dz), 'deltax': np.float32(delta1[0]), 'deltay': np.float32(delta1[1])},
                        out=g.real)
            ne.evaluate('dz/(two_s * pi_s * RR2) * (sin(k0 * RR) * one_s/RR - cos(k0*RR) * k0) * deltax * deltay',
                        {'RR': rr, 'RR2': rr2, 'k0': np.float32(2 * np.pi / self.optics.wavelength_medium),
                         'two_s': np.float32(2), 'one_s': np.float32(1), 'pi_s': np.float32(np.pi),
                         'dz': np.float32(self.dz), 'deltax': np.float32(delta1[0]), 'deltay': np.float32(delta1[1])},
                        out=g.imag)

        elif self.complexdtype == np.complex128:

            rr2 = ne.evaluate('xx_pad*xx_pad+yy_pad*yy_pad+dz*dz',
                              {'xx_pad': xx_pad, 'yy_pad': yy_pad, 'dz': np.float32(self.dz)})
            rr = ne.evaluate('sqrt(rr2)', {'rr2': rr2})
            ne.evaluate('dz/ ( 2 * pi) * exp(1j * k0 * RR) / RR2 * (1./RR -1j * k0) * deltax * deltay',
                        {'dz': self.dz, 'pi': np.pi, 'k0': 2 * np.pi / self.optics.wavelength_medium, 'RR': rr, 'RR2': rr2,
                         'deltax': delta1[0], 'deltay': delta1[1]}, out=g)

        R2_max = rr2.max()
        self.d_rho = np.sqrt(self.optics.wavelength_medium ** 2 + R2_max + 2 * self.optics.wavelength_medium * np.sqrt(
            R2_max + self.dz ** 2)) - np.sqrt(R2_max)

        if ((self.d_rho < delta1[0]) or (self.d_rho < delta1[1])):
            print('sampling condition not fulfilled, convolution kernel not sampled properly')

        self.G = np.fft.fft2(g)

        del g, rr2, rr

    def _init_work_arrays(self):
        # work arrays
        self.work = np.empty(self.N12_pad, dtype=self.complexdtype)
        self.work_F = np.empty(self.N12_pad, dtype=self.complexdtype)  ##

    def propagate(self):
        self.work.fill(0)
        self.work[:self.N1[0], :self.N1[1]] = self.field1.field

        self.work_F[:] = np.fft.fft2(self.work)

        self.work_F *= self.G

        self.work[:] = np.fft.ifft2(self.work_F)

        self.field2.field = self.work[self.N1[0]:, self.N1[1]:]

    def propagate_backward(self):
        self.work.fill(0)
        self.work[:self.N2[0], :self.N2[1]] = self.field2.field

        self.work_F[:] = np.fft.fft2(self.work)

        self.work_F *= self.G.conj()

        self.work[:] = np.fft.ifft2(self.work_F)

        self.field1.field = self.work[self.N2[0]:, self.N2[1]:]


class RayleighSommerfeldPropagatorGPU(RayleighSommerfeldPropagator):
    def __init__(self, field1, field2, optics, dz,
                 cl_context=None, cl_queue=None,
                 cl_profile=False):
        self.context = cl_context
        self.queue = cl_queue
        self.cl_profile = cl_profile
        super().__init__(field1, field2, optics, dz)

    def _init(self):
        # override parent method for custom initialization
        self._init_data()
        self._init_cl()
        self._init_cl_arrays()
        self._init_cl_fft()
        self._init_cl_kernels()

    def _init_cl(self):
        if self.queue is not None:
            self.context = self.queue.context

        if self.context is None:
            platform = cl.get_platforms()[0]
            self.device = platform.get_devices(cl.device_type.GPU)[0]
            self.context = cl.Context([self.device])

        if self.queue is None:
            props = cl.command_queue_properties.PROFILING_ENABLE if self.cl_profile else 0
            self.queue = cl.CommandQueue(self.context, properties=props)
        else:
            if self.queue.properties & cl.command_queue_properties.PROFILING_ENABLE:
                # queue profiling enabled
                if not self.cl_profile:
                    print('disabling profiling')
                    self.cl_profile = False
            else:
                # queue profiling disabled
                if self.cl_profile:
                    print('cannot enable profiling because queue not configured for profiling, disabling it')
                    self.cl_profile = False

    def _init_cl_arrays(self):

        self.cl_G = cla.to_device(self.queue, self.G.astype(self.complexdtype))
        self.cl_G_conj = cla.to_device(self.queue, self.G.astype(self.complexdtype).conj())

        self.cl_work = cla.zeros(self.queue, tuple(self.N12_pad), self.complexdtype)
        self.cl_workF = cla.zeros_like(self.cl_work)

        self.cl_field1 = cla.empty(self.queue, tuple(self.N1), self.complexdtype)
        self.cl_field2 = cla.empty(self.queue, tuple(self.N2), self.complexdtype)

    def _init_cl_fft(self):

        self.cl_fftplan = gpyfft.fft.FFT(self.context, self.queue, self.cl_work, self.cl_workF)
        self.cl_ifftplan = gpyfft.fft.FFT(self.context, self.queue, self.cl_workF, self.cl_work)

    def _init_cl_kernels(self):

        source = open('propagator_RS.cl', 'r').read()
        program = cl.Program(self.context, source)
        program.build()
        build_info = program.get_build_info(self.context.devices[0],
                                            cl.program_build_info.LOG)
        if build_info:
            print("warning: non-empty build info:")
            print(build_info)

        self.cl_kernel_premul = program.premul_field_2d
        # self.cl_kernel_premul.set_scalar_arg_dtypes(('uint32',) * 2 + (None,) * 4)

        self.cl_kernel_mul = program.multiply_workF_2d

        self.cl_kernel_postmul = program.postmul_field_2d
        # self.cl_kernel_postmul.set_scalar_arg_dtypes(('uint32',) * 2 + (None,) * 2 + ('float32',) + (None,) * 2)

    def propagate(self):

        gws12p = (int(self.N12_pad[1]),  # reverse order!!
                  int(self.N12_pad[0]))

        local_work_size = (8, 8)

        # copy field data
        cl.enqueue_copy(self.queue, self.cl_field1.data, self.field1.field)

        # prepare work array: fill in zero padded copy of field1, premultiply with w1
        self.cl_kernel_premul.set_args(
            np.uint32(self.N1[0]), np.uint32(self.N1[1]),
            self.cl_field1.data,
            self.cl_work.data,
        )

        evt1 = cl.enqueue_nd_range_kernel(
            self.queue,
            self.cl_kernel_premul,
            global_work_size=gws12p,
            local_work_size=local_work_size
        )

        # profile(evt1, 'premul')

        # forward fft of work array
        evt2, = self.cl_fftplan.enqueue()

        # profile(evt2, 'fft')

        # multiply with kernel
        self.cl_kernel_mul.set_args(
            self.cl_workF.data,
            self.cl_G.data
        )

        ###
        evt3 = cl.enqueue_nd_range_kernel(
            self.queue,
            self.cl_kernel_mul,
            global_work_size=(int(self.N12_pad[1]),  # reverse order!!
                              int(self.N12_pad[0])),
            local_work_size=local_work_size
        )

        # profile(evt3, 'mul')

        ## dbg_workF = self.cl_workF.get()

        # backward fft

        evt4, = self.cl_ifftplan.enqueue(  #
            forward=False
        )

        ## np.testing.assert_almost_equal(np.fft.ifft2(dbg_workF), self.cl_work.get(), 5)

        # extract field2, multiplied with w2
        self.cl_kernel_postmul.set_args(
            np.uint32(self.N12_pad[0]), np.uint32(self.N12_pad[1]),
            np.uint32(self.N1[0]), np.uint32(self.N1[1]),
            self.cl_work.data,
            self.cl_field2.data
        )
        evt5 = cl.enqueue_nd_range_kernel(self.queue,
                                          self.cl_kernel_postmul,
                                          global_work_size=(int(self.N2[1]),  # reverse order!!!
                                                            int(self.N2[0])),
                                          local_work_size=local_work_size)

        evt5.wait()

        # profile(evt5, 'postmul')

        # copy to numpy array
        self.field2.field = self.cl_field2.get()

    def gradient_backprop(self, field2grad):

        # assume: called (immedeately after propagate)
        # ident zu propagate, aber alles conjugiert, und field1/field2 vertauscht !?

        gws12p = (int(self.N12_pad[1]),  # reverse order!!
                  int(self.N12_pad[0]))

        local_work_size = (8, 8)

        ## copy field data # TODO: not needed, already there after callling propagate()
        # cl.enqueue_copy(self.queue, self.cl_field2.data, self.field2.field)
        cl.enqueue_copy(self.queue, self.cl_field2.data, field2grad)

        # prepare work array: fill in padded copy of field2, premultiply with w2
        self.cl_kernel_premul.set_args(
            np.uint32(self.N2[0]), np.uint32(self.N2[1]),
            self.cl_field2.data,
            self.cl_work.data,
        )

        evt1 = cl.enqueue_nd_range_kernel(
            self.queue,
            self.cl_kernel_premul,
            global_work_size=gws12p,
            local_work_size=local_work_size
        )

        # profile(evt1, 'premul')

        # forward fft of work array
        evt2, = self.cl_fftplan.enqueue(
            # forward=True,
        )

        # profile(evt2, 'fft')

        # multiply with kernel
        self.cl_kernel_mul.set_args(
            self.cl_workF.data,
            self.cl_G_conj.data
        )

        evt3 = cl.enqueue_nd_range_kernel(
            self.queue,
            self.cl_kernel_mul,
            global_work_size=gws12p,
            local_work_size=local_work_size
        )

        # profile(evt3, 'mul')

        ## dbg_workF = self.cl_workF.get()

        # backward fft
        evt4, = self.cl_ifftplan.enqueue(  #
            forward=False
        )

        ## np.testing.assert_almost_equal(np.fft.ifft2(dbg_workF), self.cl_work.get(), 5)

        # extract field1, multiplied with w1_conj

        self.cl_kernel_postmul.set_args(
            np.uint32(self.N12_pad[0]), np.uint32(self.N12_pad[1]),
            np.uint32(self.N2[0]), np.uint32(self.N2[1]),
            self.cl_work.data,
            self.cl_field1.data
        )
        evt5 = cl.enqueue_nd_range_kernel(self.queue,
                                          self.cl_kernel_postmul,
                                          global_work_size=(int(self.N1[1]),  # reverse order!!!
                                                            int(self.N1[0])),
                                          local_work_size=local_work_size)

        evt5.wait()

        # profile(evt5, 'postmul')

        field1grad = self.cl_field1.get()
        return field1grad

class AngularSpectrumPropagator(object):
    def __init__(self, field1, field2, optics, dz=1e-6):
        assert field1.dtype == field2.dtype, 'fields need to have same precision'
        assert field1.ndim == field2.ndim, 'fields need to have same rank'
        assert field1.shape == field2.shape, 'fields need to have same shape'
        assert np.all(field1.size == field2.size), 'fields need to have same size'
        assert field1.vectorial == field2.vectorial, 'fields need to have same vectorial mode'

        self.optics = optics
        self.dz = dz
        self.field1 = field1
        self.field2 = field2
        self.dtype = field1.dtype
        self.ndim = field1.ndim
        self.complexdtype = field1.complexdtype
        self.vectorial = field1.vectorial

        self._init()

    def _init(self):
        self._init_data()
        self._init_work_arrays()
        self._init_fft()

    def _init_data(self):
        # TODO padding not necessary?
        # array sizes
        N1 = np.array(self.field1.shape)
        N2 = np.array(self.field2.shape)
        #N12_pad = N1 + N2  # - 1
        N12 = N1 + N2 - 1
        N12_pad = np.array([next_regular_mul8(n12) for n12 in N12])
        self.N1 = N1
        self.N2 = N2
        self.N12_pad = N12_pad

        # build kernel
        delta1, delta2 = self.field1.delta, self.field2.delta
        if (abs(delta1 - delta1).max() > 1e-7):
            print('sampling in source field is not equal to sampling in destination field')

        # TODO   self.optics.refractive_index
        #
        kx, ky = (np.fft.fftfreq(N12_pad[i], delta1[i]) for i in (0,1))
        Kx, Ky = np.meshgrid(kx, ky, sparse=True, indexing='ij')

        Kz = np.sqrt( (1./self.optics.wavelength_medium)**2 - (Kx*Kx + Ky*Ky))
        Kz[np.isnan(Kz)] = 0  # TODO: evanescent waves? ensure imag.Kz > 0

        # TODO: abs(dz) ??
        self.G = np.exp((2j*np.pi) * Kz * self.dz).astype(self.complexdtype)

    def _init_work_arrays(self):
        # work arrays
        shape = tuple(self.N12_pad)
        if self.vectorial:
            shape = (3,) + shape
        self.work = np.empty(shape, dtype=self.complexdtype)
        #self.work_F = np.empty(shape, dtype=self.complexdtype)
        self.work_F = None

    def _init_fft(self):
        self.fft_forward = functools.partial(mkl_fft.fftn, axes=range(-self.ndim, 0), overwrite_x=True)
        self.fft_backward = functools.partial(mkl_fft.ifftn, axes=range(-self.ndim, 0), overwrite_x=True)

    def propagate(self):
        self.work.fill(0)
        self.work[..., :self.N1[0], :self.N1[1]] = self.field1.field

        self.work_F = self.fft_forward(self.work)

        self.work_F *= self.G

        self.work[:] = self.fft_backward(self.work_F)

        self.field2.field = self.work[...,:self.N1[0], :self.N1[1]]

    def propagate_backward(self):
        self.work.fill(0)
        self.work[..., :self.N2[0], :self.N2[1]] = self.field2.field

        self.work_F = self.fft_forward(self.work)

        self.work_F *= self.G.conj()

        self.work[:] = self.fft_backward(self.work_F)

        self.field1.field = self.work[...,:self.N2[0], :self.N2[1]]


def test_propagate_init():
    field1 = Field(array=np.ones((8,)), dtype=np.float32)
    field2 = Field(shape=9, dtype=np.float32)  # 7
    prop = FocusPropagator(field1, field2)
    prop.propagate()
    return prop


def test_propagate_Fresnel_init():
    field1 = Field(shape=8, dtype=np.float32)
    field2 = Field(shape=8, dtype=np.float32)
    optics = ParaxialOptics(ABCD=[[0, 1. / 3], [-3, 0]])
    p = FresnelPropagator(field1, field2, optics)
    p.propagate()
    return p


def test_FresnelGPU_init():
    shape1 = (16, 8)
    shape2 = (8, 16)

    shape1 = (512, 1024)
    shape2 = (256, 512)

    field1 = Field(shape=shape1, dtype=np.float32)
    field2 = Field(shape=shape1, dtype=np.float32)

    field1.field[:] = 0.
    field1.field[:12, :6] = 1.

    optics = ParaxialOptics(ABCD=[[0, 1. / 3], [-3., 0]])

    platform = cl.get_platforms()[0]
    device = platform.get_devices(cl.device_type.GPU)[1]
    context = cl.Context([device])

    
    p = FresnelPropagatorGPU(field1, field2, optics, cl_profile=True, cl_context=context)
    return p


def test_FresnelGPU_propagate(P):
    P.propagate()

    return P.field2.field


def test_propagate_speed():
    field1 = Field(shape=(512, 512), dtype=np.float32)
    field2 = Field(shape=(512, 512), dtype=np.float32)
    prop = FocusPropagator(field1, field2)
    prop.propagate()
    return prop


if __name__ == '__main__':
    # P = test_propagate_init()
    # P = test_propagate_speed()
    # P = test_propagate_Fresnel_init()
    #prop = test_FresnelGPU_init()
    #print(prop, prop.context)
    #E2 = test_FresnelGPU_propagate(prop)
    pass

# https://github.com/endolith/scipy/blob/czt/scipy/fftpack/czt.py
