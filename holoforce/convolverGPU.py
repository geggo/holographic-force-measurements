import attr
import numpy as np
from .cltools import CLConsumer

import pyopencl.array as cla
import gpyfft

@attr.s
class ConvolverGPU(CLConsumer):
    """convolve image (float32)"""
    x_cl = attr.ib(kw_only=True)
    k_cl = attr.ib(kw_only=True)

    X_cl = attr.ib(init=False)
    K_cl = attr.ib(init=False)

    def __attrs_post_init__(self):
        # self.mem_pool = cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.cl_queue))

        self.X_cl = self._create_transform_buffer_cl(self.x_cl)
        self.K_cl = self._create_transform_buffer_cl(self.k_cl)

        self.rfft = gpyfft.FFT(self.cl_context, self.cl_queue, self.x_cl, self.X_cl)
        self.irfft = gpyfft.FFT(self.cl_context, self.cl_queue, self.X_cl, self.x_cl, real=True)

        self.rfft.enqueue_arrays(self.k_cl, self.K_cl)
        # TODO: free self.k_cl after initialization

    def _create_transform_buffer_cl(self, x_cl):
        "return cl Array with right shape for holding Fourier transform of (real) input x_cl"
        M, N = x_cl.shape
        X_cl = cla.zeros(self.cl_queue, shape=(M, N // 2 + 1), dtype=np.complex64, allocator=self.cl_allocator)
        return X_cl

    def convolve_gpu(self, x_cl=None):
        """inplace convolution of x_cl (defaults to self.x_cl)"""
        if x_cl is None:
            x_cl = self.x_cl

        evt1, = self.rfft.enqueue_arrays(x_cl, self.X_cl)  # 500 µs
        self.X_cl *= self.K_cl  # 135 µs
        evt2, = self.irfft.enqueue_arrays(self.X_cl, x_cl, forward=False) #500 µs
        return x_cl

    def convolve(self, x):
        self.x_cl.set(x)
        self.convolve_gpu()
        return self.x_cl.get()

    def convolve_bar_gpu(self, x_cl_bar):
        self.rfft.enqueue_arrays(x_cl_bar, self.X_cl)
        self.X_cl *= self.K_cl.conj()
        self.irfft.enqueue_arrays(self.X_cl, x_cl_bar, forward=False)
        return x_cl_bar

    def convolve_bar(self, x_bar):
        self.x_cl.set(x_bar)
        self.convolve_bar_gpu(self.x_cl)
        return self.x_cl.get()
