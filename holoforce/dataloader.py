import attr
import numpy as np
import os.path


@attr.s
class DataLoader(object):
    """
    load measurement files
    """

    folder = attr.ib()  #: str : data folder path
    name = attr.ib()  #: str : measurement name

    #: pattern to create filename (from folder and measurement name)
    pattern = attr.ib()

    @pattern.default
    def _pattern_default(self):
        pattern = os.path.expanduser(self.folder) + self.name + '-%02d.npz'
        return pattern

    def load_file(self, nr):
        file = np.load(self.pattern % nr, mmap_mode='r', allow_pickle=True)
        return file

    def load_data(self, nr, idx=None, normalize_exposure=True):
        """
        load data from npz files (with fields: 'bfps' and 'slm_patterns')

        Returns
        -------

        back focal plane image (normalized for 10 ms exposure) : array
        SLM phase pattern : array
        SLM phase pattern static aberration : array
        """

        m = np.load(self.pattern % nr, mmap_mode='r', allow_pickle=True)
        bfps = m['bfps']
        Imax = 2 ** (8 * bfps.dtype.itemsize) - 1
        bfps = bfps.astype(np.float32) * (1. / Imax)

        phases = m['slm_patterns']
        phase0 = m['slm_pattern0']

        state = m['state'].item()
        exposure = state['cam_bfp.exposure']

        if normalize_exposure:
            bfps *= 10000. / exposure

        if idx is not None:
            return bfps[idx], phases[idx], phase0

        return bfps, phases, phase0

    def load_bfp(self, nr, idx, name = 'bfps'):
        """
        load data from npz files

        Returns
        -------

        back focal plane image (normalized for 10 ms exposure) 
        """

        m = np.load(self.pattern % nr, mmap_mode='r', allow_pickle=True)
        bfp = m[name][idx]
        Imax = 2 ** (8 * bfp.dtype.itemsize)
        bfp = bfp.astype(np.float32) * (1. / Imax)

        state = m['state'].item()
        exposure = state['cam_bfp.exposure']

        bfp *= 10000. / exposure

        return bfp

    def load_bg_dark_images(self, nr):
        """load dark frame images, normalized and scaled for equivalent of 10ms exposure"""
        f = np.load(self.pattern % nr, mmap_mode='r', allow_pickle=True)

        bfp_bg = f['bfps_bg']
        focal_bg = f['focal_bg']

        state = f['state_bg'].item()
        exposure_bfp = state['cam_bfp.exposure']
        exposure_focal = state['cam_focal.exposure']

        Imax = 2 ** (8 * bfp_bg.dtype.itemsize)
        bfp_bg = bfp_bg.astype(np.float32) * (1. / Imax)
        bfp_bg *= 10000. / exposure_bfp

        Imax2 = 2 ** (8 * focal_bg.dtype.itemsize)
        focal_bg = focal_bg.astype(np.float32) * (1. / Imax2)
        focal_bg *= 10000. / exposure_focal

        return bfp_bg, focal_bg

    def load_bg(self, nr):
        m = np.load(self.pattern % nr, mmap_mode='r', allow_pickle=True)
        I0 = m['image']
        # Imax = 2 ** (8 * I0.dtype.itemsize)
        Imax = 2 ** (8 * I0.dtype.itemsize)  # 2**16
        I0 = I0.astype(np.float32) * (1. / Imax)

        state = m['state'].item()
        exposure = state['cam_bfp.exposure']

        I0 *= 10000. / exposure

        return I0

    def load_trap_positions(self, nr, idx=None):
        m = np.load(self.pattern % nr, mmap_mode='r', allow_pickle=True)
        if idx == None:
            return m['trap_positions'][:, :, :3]
        else:
            return m['trap_positions'][idx, :, :3]

    def load_I0_and_focal_mask_data(self, nr):  # loading data to create mask
        # Imax = 2**16

        m = np.load(self.pattern % nr, mmap_mode='r', allow_pickle=True)
        I_mask = m['image_fourier_mask'].astype(np.float32) * (1. / (2 ** 16))
        I_spots = m['image_fourier_mask_spots'].astype(np.float32) * (1. / (2 ** 16))
        pos_spots = m['trap_pos_warp']  # [:,0:2] #only xy

        I0 = m['image_A0']  # float64
        Imax = 2**16
        I0 = I0.astype(np.float32) * (1. / Imax)

        state = m['state_A0'].item()
        exposure = state['cam_bfp.exposure']

        I0 *= 10000. / exposure
        # print(exposure)

        return I_mask, I_spots, pos_spots, I0
    
    def load_I0(self, nr):  # loading data to create mask
        # Imax = 2**16

        m = np.load(self.pattern % nr, mmap_mode='r', allow_pickle=True)
        I0 = m['image_A0']  # float64
        Imax = 2**16
        I0 = I0.astype(np.float32) * (1. / Imax)

        state = m['state'].item()
        exposure = state['cam_bfp.exposure']

        I0 *= 10000. / exposure
        # print(exposure)

        return I0

    def load_bfps_single_trap(self, nr, idx=None):  # loading data to create mask

        m = np.load(self.pattern % nr, mmap_mode='r', allow_pickle=True)
        if idx == None:
            bfps_single = m['bfps_single_trap']

            Imax = 2 ** (8 * bfps_single[0][0].dtype.itemsize)
        else:
            bfps_single = m['bfps_single_trap'][idx]
            Imax = 2 ** (8 * bfps_single[0].dtype.itemsize)
        bfps_single = bfps_single.astype(np.float32) * (1. / Imax)

        exposure = m['exposure_bfp_single_trap_measurement']

        bfps_single *= 10000. / exposure
        return bfps_single

    def load_sideport_images(self, nr, idx=None):  # loading data to create mask

        m = np.load(self.pattern % nr, mmap_mode='r', allow_pickle=True)
        if idx == None:
            focal = m['focal']

            # Imax = 2 ** (8 * focal[0].dtype.itemsize)
        else:
            focal = m['focal'][idx]
            # Imax = 2 ** (8 * focal[0].dtype.itemsize)

        focal = focal.astype(np.float32)  # * (1. / Imax)

        exposure = m['state'].item()['cam_focal.exposure']

        # focal *= 10000. / exposure
        print('exposure bfp is %02d us' % exposure)
        return focal
    
    def load_single_slm_pattern(self, nr, idx = None):
    
        m = np.load(self.pattern % nr, mmap_mode='r', allow_pickle = True)
        if idx == None:
            slm_pattern = m['slm_patterns_single']
        
        else:
            slm_pattern = m['slm_patterns_single'][idx]
            
        return slm_pattern