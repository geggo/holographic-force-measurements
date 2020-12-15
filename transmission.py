import attr
import numpy as np
import pyopencl.array as cla

from cltools import CLConsumer


def fresnel_intensity(cos_alpha, cos_beta, n1, n2):
    """
    calculate intensity transmission coefficients

    Parameters
    ----------
    cos_alpha : float
        cosine angle (relative to normal) incoming wave
    cos_beta : float
        cosine angle outgoing wave
    n1, n2 : float
        refractive indices

    Returns
    -------
    T_p, T_s : float
        intensity transmission for polarization components in-plane and normal to incident plane

    """
    n = n2/n1

    t_s = 2 * n1 * cos_alpha / (n1 * cos_alpha + n2 * cos_beta)
    t_p = 2 * n1 * cos_alpha / (n2 * cos_alpha + n1 * cos_beta)
    t_I = n * cos_beta/cos_alpha
    T_s = np.square(t_s) * t_I
    T_p = np.square(t_p) * t_I
    return T_s, T_p


@attr.s#define
class FresnelParams:
    n_air: float = 1.
    n_water: float = 1.32
    n_glass: float = 1.51


@attr.s
class TransmissionFresnelGPU(CLConsumer):
    shape: tuple = attr.ib(default=(1024, 1024))
    r0: float = attr.ib(kw_only=True)
    params: FresnelParams = attr.ib(factory=FresnelParams, kw_only=True)

    transmission: np.ndarray = attr.ib(init=False, repr=False)
    cl_transmission: cla.Array = attr.ib(init=False, repr=False)

    @transmission.default
    def _transmission_default(self):
        return self._init_transmission(r0=self.r0, **attr.asdict(self.params))

    @cl_transmission.default
    def cl_transmission_default(self):
        return cla.to_device(ary=self.transmission.astype(np.float32),
                             queue=self.cl_queue, allocator=self.cl_allocator)

    def apply(self, ary):
        return ary * self.transmission

    def apply_gpu(self, ary):
        return ary * self.cl_transmission

    def apply_inline_gpu(self, ary):
        ary *= self.cl_transmission

    def compensate(self, ary, clip=0.1): #not really needed here though? 
        """

        Parameters
        ----------
        ary
        clip : float


        Returns
        -------
        ary * 1./transmission (limited to 1./clip)
        """
        pass

         
    def _init_transmission(self, r0,
                           n_air=1, n_water=1.32, n_glass=1.51):
        nx, ny = self.shape
        x, y = np.linspace(-1, 1, nx), np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y, sparse=True, indexing='xy')
        R = np.hypot(X, Y)
        cos_phi, sin_phi = X/R, Y/R
        sin_theta = R / r0

        # water/glass interface
        sin_theta_g = sin_theta * n_water / n_glass
        T_s_wg, T_p_wg = fresnel_intensity(cos_alpha=np.sqrt(1 - np.square(sin_theta)),
                                           cos_beta =np.sqrt(1 - np.square(sin_theta_g)),
                                           n1=n_water, n2=n_glass)


        # measured transmission of condensor described by 
        tp = 0.
        ts = 0.21
        # front lens/air interface
        T_cond_p = (1 - (X/r0)*(X/r0) * tp) #maske * ((x*x + y*y < 1.f)
        T_cond_s =  (1 - (Y/r0)*(Y/r0) * ts) #maske * ((x*x + y*y < 1.f)
        
        T_cond = (np.square(cos_phi) * T_cond_p   + np.square(sin_phi) * T_cond_s) * np.where(R<r0, 1,0)

        T_wg   = np.square(cos_phi) * T_p_wg   + np.square(sin_phi) * T_s_wg
        
        T = np.transpose(T_cond * T_wg)
        np.nan_to_num(T, copy=False)
        return T #, T_wg, T_cond
    
    
    


