from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

class Field(object):
    """
    Represent field (with metadata)

    Parameters
    ----------
    array : np.ndarray, optional
       array data
    shape : tuple of ints, optional
       shape of field data, single int allowed (for 1D array), Note: internally shape is
       (3,)+shape for vectorial fields
    dtype : dtype,  optional
       dtype of field data, if omitted, take from array, or use np.float32 as default      
    size : arraylike
       spatial extent covered by field data
    center : arraylike
       origin (center) of field data
    vectorial : bool, optional
       scalar or vectorial, default False
    r0 : float, optional
       pupil radius, which corresponds to light at right angle

    """

    def __init__(self,
                 shape=None,
                 array = None,
                 dtype=None,
                 size=None, center=None,
                 vectorial = False,
                 r0 = 0.5):
        
        self.vectorial = vectorial #TODO: find better way to store vectorial (field_dtype ??)

        if array is not None:
            assert shape is None
            shape = array.shape
            if vectorial:
                shape = shape[1:]  # vectorial array
            if dtype is None:
                dtype = array.real.dtype
                        
        if shape is None and array is None:
            shape = (101, 101)
        if isinstance(shape, int):
            shape = (shape,)

        #: array shape of field data
        self.shape = shape
        #: rank
        self.ndim = len(shape)

        if size is None:
            size = (1.,)*self.ndim
        
        if center is None:
            center = (0.,)*self.ndim

        assert len(size) == self.ndim
        assert len(center) == self.ndim

        if dtype is None:
            dtype = np.float32
        self.dtype = np.dtype(dtype)

        assert np.issubdtype(dtype, np.floating), "dtype needs to be of floating type"
        
        self.complexdtype = np.result_type(dtype, 1j) #dtype for complex field
        #self.field_dtype = np.dtype( (self.complexdtype, (3,)), ) if vectorial else self.complexdtype
        
        #: size spatial extent field data, size 1: -0.5 ... 0.5
        #assumes: first/last data point at center +- size/2 for each axis
        self.size = np.asarray(size, dtype=self.dtype) 
        #: center position of field data, currently ignored
        self.center = np.asarray(center, dtype=self.dtype)

        #: for pupil field: radius which corresponds to light at right angle
        self.r0 = r0
        
        #self.delta = tuple( (s/(n-1)) for s,n in zip(self.size, self.shape) ) #spacing between two grid points
        #fails for axis length 1
        self.delta = self.size / (np.array(self.shape)-1)
        
        ##: list of integer indices for data grid, sparse, with same dtype
        #self.idx = [np.arange(n, dtype=self.dtype) for n in self.shape]

        #:list of grid positions (coordinate vector)
        self.x = [np.linspace(self.center[k]-0.5*self.size[k],
                              self.center[k]+0.5*self.size[k],
                              self.shape[k],
                              dtype = self.dtype)
                  for k in range(self.ndim)  ]
        
        #: sparse coordinate matrices for grid positions, e.g. shape (n, 1, 1) for first axis
        self.X = np.meshgrid(*self.x, sparse=True, copy=False, indexing='ij')

        if array is not None:
            self._field = np.asarray(array, dtype=self.complexdtype)
        else:
            self._init_field()
            self.field = 0.

    def _init_field(self):
        "initialize private field data storage"
        shape = (3,) + self.shape if self.vectorial else self.shape    
        self._field = np.empty(shape, dtype = self.complexdtype)
        
    @property
    def field(self):
        "access field data (as ndarray)"
        return self._field

    @field.setter
    def field(self, value):
        """set field data. check for size and dtype, cast if necessary"""
        value = np.asarray(value, dtype=self.complexdtype)
        self._field[:] = value

    #TODO: write setter for delta (grid_spacing) ????

    def __repr__(self):
        return "{ndim}D Field( shape={shape}, size={size}, center={center}, dtype={complexdtype}, r0={r0})".format(**self.__dict__)

    @property
    def intensity(self):
        f = self.field
        I = np.square(f.real) + np.square(f.imag)
        if self.vectorial:
            #TODO: wrong, calculate Poynting vector (from E _and_ B)
            I = I.sum(axis=0)
        return I

    @property
    def power(self):
        P = np.prod(self.delta)*self.intensity.sum()
        # TODO: P1 = trapz(trapz(p1.intensity, x=p1.x[0], axis=0), x=p1.x[1], axis=0)
        return P

    def get_extent(self):
        x, y = self.x[0], self.x[1]
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        extent = (x[0]-0.5*dx, x[-1]+0.5*dx,
                  y[-1]+.5*dy, y[0]-0.5*dy)
        return extent

    def plot_intensity(self, ax=None, mask=None, **kwargs): #TODO: vmax arguments
        kwargs = kwargs.copy()
        if ax is None:
            ax = plt.gca()
        if self.ndim==1:
            ax.plot(self.x[0], self.intensity, **kwargs)
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter(unit='m'))
        elif self.ndim==2:
            kwargs.setdefault('cmap', plt.cm.gray)
            kwargs.setdefault('interpolation', 'nearest') #'lanczos'
            kwargs.setdefault('resample', True)
            I = self.intensity.copy()
            if mask is not None:
                I[~mask] = np.NaN
            ax.imshow(I.T,
                      extent=self.get_extent(),
                      aspect = 'equal',
                      **kwargs
                      )
            ax.axis('image')
            ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter(unit='m'))
            ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter(unit='m'))

    def _plot_phase(self, field, ax=None, **kwargs):
        kwargs = kwargs.copy()
        """
        Plot phase of single component

        Parameters
        ----------
        field : 2D array 
        """
        
        if ax is None:
            ax = plt.gca()
        img = plt.cm.hsv( (np.angle(field.T)+np.pi)/(2*np.pi) )  # apply cyclic colormap
        intens = field.real*field.real + field.imag*field.imag
        intens = intens.T
        intens_max = intens.max()
        intens *= (1.0/intens_max)
        img[:,:,3] = np.clip(2*intens, 0.0, 1.0)
        kwargs.setdefault('interpolation', 'lanczos')
        ax.imshow(img,
                  extent=self.get_extent(),
                  aspect = 'equal',
                  **kwargs
        )
        ax.xaxis.set_major_formatter(mpl.ticker.EngFormatter(unit='m'))
        ax.yaxis.set_major_formatter(mpl.ticker.EngFormatter(unit='m'))

    def plot_field(self, **kwargs):
        field = self.field[0] if self.vectorial else self.field
        fig, (ax0, ax1) = plt.subplots(1,2)
        self.plot_intensity(ax=ax0, **kwargs)

        self._plot_phase(field, ax=ax1, **kwargs)
        return ax0, ax1
        
    def __getitem__(self, key):
        if isinstance(key, int): # convert single integer to tuple
            key = (key,)
        assert isinstance(key, tuple)

        # append slice(None, None) for missing dimensions
        if len(key) < self.ndim:
            key = key + (slice(None),)*(self.ndim - len(key))

        if self.vectorial:
            key = (Ellipsis,) + key  # prepend ... for vectorial fields
        subarray = self.field[key]
        subcoords = [self.x[k][key[k]] for k in range(self.ndim)]
        center = [0.5*(x[-1] + x[0]) for x in subcoords]
        size = [x[-1] - x[0] for x in subcoords]

        F = Field(array=subarray,
                  size=size,
                  center=center,
                  vectorial=self.vectorial
        )
        return F

    def copy(self):
        array = self.field.copy()
        return Field(array=array,
                     size=self.size,
                     center=self.center,
                     vectorial=self.vectorial,
                     r0=self.r0)


    def find_idx_from_pos(self, pos):
        pos = np.asanyarray(pos)
        #idx = (pos - (self.center - 0.5*self.size)) // self.delta
        idx = np.round((pos - (self.center - 0.5*self.size)) / self.delta)
        return idx.astype(np.int_)
        #TODO: clip to shape
    
class VectorField(Field):
    """
    represent vectorial field data (3 components)
    """

    
    # def _init_field(self):
    #     #self._field = np.zeros(self.shape+(3,), dtype = self.complexdtype)
    #     #TODO: unaligned components, slow down?, better: (3,)+shape? or different memory order?

    #     #different memory layout: contigous blocks for Ex, Ey, Ez
    #     #self._field = np.rollaxis(np.zeros((3,)+self.shape, dtype=self.complexdtype), 0, self.ndim+1)

    #     #broadcast friendly layout
    #     self._field = np.zeros((3,) + self.shape, dtype=self.complexdtype)
    
    def __repr__(self):
        #TODO: dtype.__name__ -> e.g. float32
        return "{ndim}D VectorField( shape={shape}, size={size}, center={center}, dtype={dtype})".format(**self.__dict__)


def test_Field():

    F = Field( shape=(11, 3))
    return F
    
