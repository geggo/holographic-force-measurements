import numpy as np

bits_to_dtype = {8: np.uint8,
                 16: np.uint16}

class CamImage(object):
    """
    encapsulates camera image

    store only ROI data, create full image on demand
    """

    w, h = 640, 480
    c = 1  # number of components
    bits = 8  # bitdepth (per component)
    dtype = np.uint8
    pos = (0,0)  # xy-position (column, row) of ROI origin
    roi_w, roi_h = 640, 480

    _fill_value = 0

    # TODO: use metadata dict instead
    timestamp = 0
    frameNr = 0
    exposure_us = 0
    gain_dB = 0.0

    def __init__(self, data=None, w=None, h=None, c=1, bits=8):
        
        self.w = 640 if w is None else w
        self.h = 480 if h is None else h
        self.c = c
        self.bits = bits
        self.dtype = bits_to_dtype[self.bits]
        self.pos = (0,0)

        if data is not None:
            self.data_roi = data  # data_roi is property, side effects on self.dtype !!!!
            self.w = w if w is not None else self.roi_w
            self.h = h if h is not None else self.roi_h
        else:
            self.data_roi = np.zeros(shape=(64, 64) if self.c == 1 else (64, 64, self.c),
                                     dtype=self.dtype)

    @property
    def data_fullsize(self):
        w, h, c = self.w, self.h, self.c
        data = np.empty((h, w) if c == 1 else (h, w, c),
                        dtype=self.dtype)
        data.fill(self._fill_value)

        c0, r0 = self.pos
        nr, nc = self.roi_h, self.roi_w
        data[r0:r0+nr, c0:c0+nc] = self._data_roi
        return data

    @property
    def data_roi(self):
        return self._data_roi

    @data_roi.setter
    def data_roi(self, value):
        self._data_roi = np.atleast_2d(value)
        self.roi_h, self.roi_w = self._data_roi.shape[:2]
        self.dtype = self._data_roi.dtype

        if len(value.dtype)>1:
            self.c = len(value.dtype)
            self.dtype = value.dtype[0]  # assume all entries have same dtype
            self._data_roi = self._data_roi.view((self.dtype, self.c))
            #if self.c == 4:
            #    self._data_roi[...,-1] = 255
        else:
            self.c = value.shape[2] if value.ndim==3 else 1
        self.bits = self._data_roi.itemsize*8


    def set_data_roi_and_pos(self, value, pos):
        self.pos = pos
        self.data_roi = value
    
if __name__ == '__main__':
    img = CamImage()
    img._fill_value = 50

    a = np.zeros((100, 100), dtype=np.uint16)

    img.data_roi = a
    data = img.data_fullsize