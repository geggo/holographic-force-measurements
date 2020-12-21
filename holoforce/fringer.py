# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
import pyopencl as cl
import pyopencl.array as cla
import os.path


class CLProfiler:
    def __init__(self, name=''):
        self.name = name
        self.log = []
        
    def profile(self, event, name=''):
        return event

    def reset(self):
        self.log = []

    def summary(self):
        return self.name


class LoggingCLProfiler(CLProfiler):
        
    def profile(self, event, name=''):
        try:
            event.wait()
            dt = 1e-6*(event.profile.end - event.profile.start)
        except cl.RuntimeError as e:
            print(e)
            dt = np.NaN

        self.log.append((name, dt))
        return event

    def summary(self):
        d = np.array([duration for (name, duration) in self.log])
        if len(d) == 0:
            d = np.zeros((1,))
                        
        return "%20s N: %d, avg: %5.2f ms, med: %5.2f ms, min: %5.2f ms"%(
            self.name,
            len(d),
            np.mean(d),
            np.median(d),
            d.min(),
            )

def E_n(x, n):
    xachse = np.linspace(-10, 10, 1001)
    k = np.exp(-np.abs(xachse)**n)*(xachse<0) + np.exp(-np.abs(xachse)**n)*(xachse>=0)
    Intk = np.cumsum(k)
    Intk *= 1./Intk.max()
    y = np.interp(x, xachse, Intk)
    return y

#default_model_datafile = 'modelfitparams_d425_2n.npz'
#default_model_datafile = 'modelfitparams_berreman_12_07_2018.npz'
#default_model_datafile = 'modelfitparams_berreman_29_10_2018.npz'
default_model_datafile = 'modelfitparams_berreman_20_11_2018.npz'


class FringerGPU(object):
    
    def __init__(self, cl_context=None, cl_queue=None, oversample = 16, profile=False, tabledatafile=None, simple=False):
        self.oversample = oversample
        self.context = cl_context
        self.queue = cl_queue
        self.profile = profile
        self.simple = simple
        self._init_cl()
        self.tabledatafile=default_model_datafile if tabledatafile is None else tabledatafile
        self.prepare_param_tables()
        self._init_cl_kernels()
        
        if profile:
            self.profiler = LoggingCLProfiler('FringerGPU')
        else:
            self.profiler = CLProfiler('FringerGPU')

        
    def _init_cl(self):
        if self.context is None:
            platform = cl.get_platforms()[0]
            self.device = platform.get_devices(cl.device_type.GPU)[0]
            self.context = cl.Context([self.device])

        if self.queue is None:
            props = cl.command_queue_properties.PROFILING_ENABLE if self.profile else 0
            self.queue = cl.CommandQueue(self.context, properties=props)

        
    def _init_cl_kernels(self):
        filename = os.path.join(os.path.split(__file__)[0],
                                'fringer.cl')
        source = open(filename, 'r').read()
        program = cl.Program(self.context, source)
        defines = ['-DPHI_MIN=%ff'%self.phases.min(),
                   '-DPHI_MAX=%ff'%self.phases.max()]
        program.build(defines)
        build_info = program.get_build_info(self.context.devices[0],
                                            cl.program_build_info.LOG)
        if build_info:
            print("warning: non-empty build info:")
            print(build_info)
            
        self.cl_kernel_fringe1d = program.fringe1d
        self.cl_kernel_fringe1d_grad = program.fringe1d_grad
        self.cl_kernel_fringe2d = program.fringe2dV2
        self.cl_kernel_fringe2d_grad = program.fringe2dV2_grad
        
    def prepare_param_tables(self):
        filename = os.path.join(os.path.split(__file__)[0],
                                self.tabledatafile)
        d = np.load(filename)
        self.phases = d['coeffs_phase']
        Tx = self.preprocess_params(d['coeffs_x'])
        Ty = self.preprocess_params(d['coeffs_y'])

        if self.simple:
            Tx[:] = np.array([0., 2.65, 2.65, 1 ])[:,None,None] #x0, xi_p, xi_m, n 2.6/1 bzw. 3./1.1
            Ty[:] = np.array([0, 2., 2., 1.])[:,None,None]
        self.cl_params_x = cl.image_from_array(self.context, np.ascontiguousarray(Tx.T, np.float32), num_channels=4)
        self.cl_params_y = cl.image_from_array(self.context, np.ascontiguousarray(Ty.T, np.float32), num_channels=4)
        self.cl_table_E = cl.image_from_array(self.context, np.ascontiguousarray(self.calc_E().T, np.float32), num_channels=1)
        
        #self.LUT_phase = d['U_phase']
        #self.LUT_U = d['U']
        
    def preprocess_params(self, C):
        """
        combine params for up/down into single array, fill in diagonals by interpolation
        """
        N = len(self.phases)
        T = np.zeros((4, N, N), np.float32)
        for idx in range(4):
            Tp = C[idx+4].copy()
            Tm = C[[0, 1, 2, 3][idx]].T.copy()

            Tpm = Tp
            selm = np.isfinite(Tm)
            Tpm[selm] = Tm[selm]
            #print np.shape(Tp),np.shape(Tm),np.shape(Tpm)
            #if idx>0:
            #    Tpm = 1./Tpm

            ridx = np.array([-1,0,1,0])
            cidx = np.array([0,-1,0,1])
            T[idx] = Tpm
            for k in range(1,N-1): #TODO: special treatment k==0 ?
                T[idx,k,k] = np.nanmean(Tpm[(ridx+k,cidx+k)])
            T[idx,0,0] = np.nanmean(Tpm[[0,1], [1,0]])
            T[idx,-1,-1] = np.nanmean(Tpm[[-2,-1],[-1,-2]])


        return T

    def calc_E(self):
        x = np.linspace(-10, 10, 501) # TODO: magic constants
        n = np.linspace(1, 2, 51)
        E = np.zeros( (len(n), len(x)) )
        for k, nk in enumerate(n):
            E[k] = E_n(x, nk)

        return E

    def fringe1d(self, pattern, direction = 'x'):
        N, = pattern.shape
        os = self.oversample

        pattern_cl = cla.to_device(self.queue, pattern.astype(np.float32))
        pattern_s_cl = cla.zeros(self.queue, (os*N,), dtype=np.float32)
        self.cl_kernel_fringe1d.set_args(
            pattern_cl.data,
            pattern_s_cl.data,
            self.cl_params_x if direction == 'x' else self.cl_params_y,
            self.cl_table_E,
        )
        evt = cl.enqueue_nd_range_kernel(
            self.queue,
            self.cl_kernel_fringe1d,
            global_work_size = (os*N,),
            local_work_size = (os,)
            )
        return pattern_s_cl.get()

    def fringe1d_grad(self, grad_phi_os, phi, direction = 'x'):
        N, = phi.shape
        os = self.oversample
        phi_cl = cla.to_device(self.queue, phi.astype(np.float32))
        grad_phi_os_cl = cla.to_device(self.queue, grad_phi_os.astype(np.float32))
        grad_phi_cl = cla.zeros(self.queue, (N,), dtype=np.float32)
        self.cl_kernel_fringe1d_grad.set_args(
            grad_phi_cl.data,
            phi_cl.data,
            grad_phi_os_cl.data,
            self.cl_params_x if direction == 'x' else self.cl_params_y,
            self.cl_table_E,
        )
        evt = cl.enqueue_nd_range_kernel(
            self.queue,
            self.cl_kernel_fringe1d_grad,
            global_work_size = (os*N,),
            local_work_size = (os,),
        )
        return grad_phi_cl.get()

    
    def fringe2d(self, pattern, debug=0):
        M,N = pattern.shape
        os = self.oversample

        pattern_cl = cla.to_device(self.queue, pattern.astype(np.float32))
        pattern_s_cl = cla.zeros(self.queue, (os*M, os*N), dtype=np.float32)
        
        self.cl_kernel_fringe2d.set_args(
            pattern_cl.data,
            pattern_s_cl.data,
            self.cl_params_x,
            self.cl_params_y,
            self.cl_table_E,
            np.int32(debug),
            )
        
        evt = cl.enqueue_nd_range_kernel(
            self.queue,
            self.cl_kernel_fringe2d,
            global_work_size = (os*N, os*M),
            local_work_size = (os, os))

        self.profiler.profile(evt, name=self.cl_kernel_fringe2d.function_name)        
        
        return pattern_s_cl.get()

    def fringe2d_grad(self, grad_phi_os, phi, debug=0):
        M,N = phi.shape
        os = self.oversample

        phi_cl = cla.to_device(self.queue, phi.astype(np.float32))
        grad_phi_os_cl = cla.to_device(self.queue, grad_phi_os.astype(np.float32))
        grad_phi_cl = cla.zeros(self.queue, (M, N), dtype=np.float32)

        self.cl_kernel_fringe2d_grad.set_args(
            grad_phi_cl.data,
            phi_cl.data,
            grad_phi_os_cl.data,
            self.cl_params_x, self.cl_params_y,
            self.cl_table_E,
            np.int32(debug),
        )
        evt = cl.enqueue_nd_range_kernel(
            self.queue,
            self.cl_kernel_fringe2d_grad,
            global_work_size = (os*N, os*M),
            local_work_size = (os, os))

        self.profiler.profile(evt, name='fringe2dgrad')

        return grad_phi_cl.get()

        
if __name__ == '__main__':
    fringer = FringerGPU()
    pattern = np.random.random((8,16)) + 0.35
    pattern_smooth = fringer.fringe2d(pattern)
    
