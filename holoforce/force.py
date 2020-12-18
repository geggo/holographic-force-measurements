import numpy as np

class CalculateForce(object):
    """
    Calcuate force from bfp images

    Parameters
    ----------
    R0 ... maximal radius of light transmitted to the BFP
    """
    def __init__(self, R0 = 1.02, bfp_fields_shape = (1024,1024), scale_au_to_pN = 1/45000):
        self.scale_au_to_pN = scale_au_to_pN #measured value, can change when detection system is changed
        
         #initialize coordinates!
        self._X, self._Y = np.linspace(-1,1,bfp_fields_shape[0]), np.linspace(-1,1,bfp_fields_shape[1])
        self._X.shape = (1,-1)
        self._Y.shape = (-1,1)
        self._R0 = R0 
            
        self._wz = np.sqrt(1-(self._X*self._X+self._Y*self._Y)/(self._R0*self._R0))
        self._wz[np.isnan(self._wz)] = 0
        self.mask_bfp = np.where((self._X**2+self._Y**2)<self._R0**2,1,0)
        
    def calc_force(self,intensity):
        intensity *= self.mask_bfp
        F = [np.sum(intensity*self._X/self._R0), np.sum(intensity*self._Y/self._R0), np.sum(intensity*self._wz)]
        return np.array(F)*self.scale_au_to_pN

if __name__ == '__main__':  
    pass
