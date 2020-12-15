import numpy as np
import pylab as plt

precision = 'single'
#precision = 'double'

class Plane(object):

    def __init__(self, w, h, Nx=1024, Ny=1024, precision = precision):
        self.precision = precision
        if precision == 'single':
            self.floattype = np.float32
            self.complextype = np.complex64
            
        elif precision == 'double':
            self.floattype = np.float64
            self.complextype = np.complex128
        
        self.w, self.h = w, h
        self.Nx = Nx
        self.Ny = Ny
        #self.N = np.array((self.Nx, self.Ny))
        
        self.dx = w/Nx #pixel size
        self.dy = h/Ny
        
        self.nx = np.arange(Nx, dtype = self.floattype) #pixel numbers
        self.nx.shape = (-1,1)
        
        self.ny = np.arange(Ny, dtype = self.floattype)
        self.ny.shape = (1,-1)

        self.nxc = self.nx - Nx/2
        self.nyc = self.ny - Ny/2

        self.x, self.y = (self.nx - Nx/2) * self.dx, (self.ny - Ny/2) * self.dy #centered pixel positions 

        self._field = np.zeros( (Nx, Ny), dtype = self.complextype)

        self.amplitude = None

    @property
    def field(self):
        return self._field

    @field.setter
    def field(self, value):
        if value.shape != (self.Nx, self.Ny):
            raise ValueError('wrong array size')
        if value.dtype == self.complextype:
            self._field = value
        else:
            print "warning: casting 'field' to", self.complextype
            self._field = value.astype(self.complextype)
            
    @property
    def intensity(self):
        return (self.field.real*self.field.real + self.field.imag*self.field.imag)
   
    @intensity.setter
    def intensity(self, intensity):
        self.amplitude = np.sqrt(intensity)
 
    def plot_intensity(self, field = None):
        w, h = 1e3*self.w, 1e3*self.h
        intensity = self.intensity if field is None else np.abs(field)**2
        plt.imshow(intensity.T, 
                   extent = (-0.5*w, 0.5*w, -0.5*h, 0.5*h), 
                   aspect = 'equal', 
                   #interpolation = 'nearest',
                   #interpolation = 'bicubic',
                   interpolation = 'sinc',
                   resample = True,
                   cmap = plt.cm.gray,
                   hold = False)
        
    def plot_phase(self, field = None):
        field = self.field if field is None else field
        w, h = 1e3*self.w, 1e3*self.h
        img = plt.cm.hsv( (np.angle(field.T)+np.pi)/(2*np.pi) )
        #intens = self.intensity.T
        intens = field.real*field.real + field.imag*field.imag
        intens = intens.T
        intens *= (1.0/intens.max())
        img[:,:,3] = intens
        plt.imshow(img,
                   extent = (-0.5*w, 0.5*w, -0.5*h, 0.5*h), 
                   aspect = 'equal', 
                   interpolation = 'nearest',
                   hold = False)
