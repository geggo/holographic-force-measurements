# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import measurements
from skimage import transform as tf
import attr

import warnings

def lowpass3(pat_hires, sigma=(0.12, 0.12), gamma = 0.5):
    "lowpass filter: convolution with gaussian kernel"
    M, N = pat_hires.shape
    x = np.fft.ifftshift(np.arange(-M/2, M/2))
    y = np.fft.ifftshift(np.arange(-N/2, N/2))
    x.shape = (-1,1)
    y.shape = (1,-1)
    sigmay, sigmax = sigma
    g = np.exp ( - (x**2/(2*(sigmax)**2) + y**2/(2*(sigmay)**2)) ** gamma)
    g *= 1./g.sum()
    pat_hires_smooth = np.fft.ifft2(np.fft.fft2(pat_hires) * np.fft.fft2(g))
    return abs(pat_hires_smooth)

def transposer(x):
    "return transposed copy of input array"
    return x.T.copy()

def convert_sideport_img(x):
    "return copy of sideport image with flipped column, normalized by cam_bit (2**16) and as a float32"
    cam_bit = 2**16
    return (x[::-1,:].astype(np.float32)/cam_bit).copy()


@attr.s
class CreateMaskFocal(object):
    """
    Create focal mask from experimental data for field/phase retrieval 

    Parameters
    ----------
    img_spots: np.ndarray 
        sideport cam image of spots for warping
    img_mask: np.ndarray
        sideport cam image of random pattern, substracting the background might be necessary
    pos_spots_holo: list
        positions of traps from holo.py
    
    
    field_shape : tuple
       shape of destination field (default = (1024,1024))
    field_size : np.ndarray
       size of destination field (default=None ), if default is None the size is calculated automatically 
    field_center : tuple 
       center of destination field (default = np.array(( 0, 0)))
    s: float
        scaling for coordinates from holo.py to field (default: 1e-6 * 60 * 300 / 180 * 1.17)
    cam_bit: float
        camera/saved image setting bitdepth (default: 2**16)
        
    dxy_mask_img : float
        scaling for calculating field_size automatically with diameter of untransformed mask,
        determined experimentally, might need to change it slightly (its based on two measurements at the moment)
    
    mask: np.ndarray
        created mask
    """
    
    mask = attr.ib(default = None, init = False)
    dxy_mask_img = attr.ib(default = 3.95e-5, kw_only = False)
    
    
    field_shape: tuple = attr.ib(kw_only = True, default=( 1024, 1024))
    field_size: tuple  = attr.ib(kw_only = False, default=None)#np.array((15e-3,) * 2 ))
    field_center: tuple  = attr.ib(kw_only = True, default = np.array(( 0, 0)) )

                          
    s: float = attr.ib(kw_only = True, default = 1e-6 * 60 * 300 / 180 * 1.17) 
    #cam_bit: int = attr.ib(kw_only = True, default = 2**16) 
    
    img_spots: np.ndarray = attr.ib(kw_only=True, converter = convert_sideport_img)
    img_mask: np.ndarray = attr.ib(kw_only=True, converter = convert_sideport_img)
    pos_spots_holo: list = attr.ib(kw_only=True)
    
    
    def _check_data(self):
        "check data"
        if self.img_spots.shape != self.img_mask.shape:
            warnings.warn('Spots and mask image have different shapes!.', Warning)

    def _create_mask_from_untransformed_img_mask(self, clip_max = 0.95, sigma=(0.4, 0.4), gamma = 0.8):
        """
        Create mask from warped img_mask 
    
        Parameters
        ----------
        sigma: Tuple
            setting for lowpass filter, defualt = (0.4, 0.4)
        gamma: Float
            setting for lowpass filter, default = 0.8
        clip_max: Float
            setting the maximal value for clipping the mask
        """
        img = self.img_mask
        
        I = lowpass3(img, sigma=sigma, gamma = gamma)
        self.mask_untransformed = np.zeros_like(I)
        I_max = I.max()
        
        #nearest neighbour = 8 pixels
        for r in range(I.shape[0]-2):
            for c in range(I.shape[1]-2):
                I_sub_sum = np.sum(I[r-1:r+2,c-1:c+2])/9
                #print(I_sub_sum)
                if I_sub_sum>I_max*0.20:#25:
                    self.mask_untransformed[r,c] = 1
                        

        self.mask_untransformed.clip(0, clip_max)
        
    def _find_fourier_plane_size(self):
        "find fouerier plane size from the untransformed mask, using the scaling dxy_mask_img"
        '''
        only do this if it is not given as keyword
        '''
        if self.field_size is None:
            sum_r, sum_c = np.sum(self.mask_untransformed, axis = 0), np.sum(self.mask_untransformed, axis = 1)
            mask_width = np.max([sum_r, sum_c])
            
            self.field_size = (self.dxy_mask_img * mask_width * 1.05, )*2 #5% larger
            
        
    def _identify_spots(self, clip_val = 0.7, sig = 0.8, gam = 0.4):
        "find spots in sideport camimage img_spots"
        '''
        can set the parameters of the lowpass as well as the clipping value if need be
        '''
        img_spots_lp = lowpass3(self.img_spots, sigma=(sig, sig), gamma = gam)
        img_spots_clip = clip_val*img_spots_lp.max() < img_spots_lp
        
        #find spots
        lw, num = measurements.label(img_spots_clip)
        
        if num > 4:
            warnings.warn('Found too many spots, check if still created the mask correctly.', Warning)
        if num < 4:
            warnings.warn('Found too few spots, check if still created the mask correctly (probably not?).', Warning)
            
        self.pos_spots_img_idx = measurements.center_of_mass(self.img_spots+0.1, labels=lw, index = np.arange(lw.max())+1)
        self.pos_spots_img_idx = [t[::-1] for t in self.pos_spots_img_idx]

        
    def _create_idx_from_pos_spots_holo(self):
        "convert pos_holo_spots to corresponding indices in field"
        self.field_delta = np.array(self.field_size) / (np.array(self.field_shape) - 1 )
        if self.field_delta[0] != self.field_delta[1]:
            warnings.warn('Focal field has different delta in x and y.', Warning)

        pos = np.asanyarray(-self.pos_spots_holo[:,0:2]*self.s) #only x,y positions
        
        self.pos_spots_holo_idx = list((pos - (np.array(self.field_center)*self.s - 0.5*np.array(self.field_size))) 
                                    / self.field_delta[0])
        
        self.pos_spots_holo_idx = [t[::-1] for t in self.pos_spots_holo_idx]
        
    def _sort_indizes_to_match(self, pm = 1):
        "match order of indices from img_spots and pos_holo_spots using sort, sensitive to changes in data!"
        '''
        pm ... set this to +-1 to change sorting (check in plot_check_sorting plot)
        '''
        self.pos_spots_img_idx_sorted = self.pos_spots_img_idx.copy()
        self.pos_spots_holo_idx_sorted = self.pos_spots_holo_idx.copy()
        
        self.pos_spots_img_idx_sorted.sort(key = lambda x: (x[0], x[1]))
        #depens on how the data is, change_sorting = +/-1
        self.pos_spots_holo_idx_sorted.sort(key = lambda x: (x[0], pm*x[1]))
    

    def _find_transformation_matrix(self):#, set_pm = True, pm = 1):   
        "calculate transformation matrix using skimage.transform.SimilarityTransform"         
        self.t_warping = tf.SimilarityTransform()
        
        pms = [1,-1] #so far this is enough to find correct way of sorting, might adapt this to go throug all possibilites not only dimension two for pos spots holo
        
        #find correct orientation
        sum_difference_transformed_coordinates = []
        for pm in pms:
            self._sort_indizes_to_match(pm = pm)
            self.t_warping.estimate(dst = np.array(self.pos_spots_img_idx_sorted) , src = np.array(self.pos_spots_holo_idx_sorted)) 
            tmp = np.sum(np.abs(np.squeeze(np.array([self.t_warping(val) for val in np.array(self.pos_spots_holo_idx_sorted)]))-np.array(self.pos_spots_img_idx_sorted)))
            sum_difference_transformed_coordinates.append(tmp)

        idx_min_sum = np.argmin(sum_difference_transformed_coordinates)
        self._sort_indizes_to_match(pm = pms[idx_min_sum])
        self.t_warping.estimate(dst = np.array(self.pos_spots_img_idx_sorted) , src = np.array(self.pos_spots_holo_idx_sorted)) 
        
    
    def _warp_mask_and_spot_data(self):
        "warp img_mask"
        self.img_spots_warped = tf.warp(self.img_spots, self.t_warping, output_shape = self.field_shape)
        self.mask = (tf.warp(self.mask_untransformed, self.t_warping, output_shape = self.field_shape)).T
        
        #check if mask covers more of the array than in the original image (is a indicator that the field size is wrong, but not necessarily so)
        if (self.mask.sum()/(self.mask.shape[0]*self.mask.shape[1])) < (self.mask_untransformed.sum()/(self.mask_untransformed.shape[0]*self.mask_untransformed.shape[1])):
            warnings.warn('Fourier_field_size might me wrong! Check resulting mask!', Warning)
           

    def create_mask(self, sigma = 0.4, gamma = 0.8, dpi = 80, figsize = (20,5)):
        self._check_data()
        self._create_mask_from_untransformed_img_mask(clip_max = 0.95, sigma=(sigma, sigma), gamma = gamma)
        self._find_fourier_plane_size()
        self._identify_spots(clip_val = 0.7, sig = 0.8, gam = 0.4)  #with standard values
        self._create_idx_from_pos_spots_holo()
        self._find_transformation_matrix()
        self._warp_mask_and_spot_data()
        self.plot_check_create_mask(dpi = dpi, figsize = figsize)
        
        
    def plot_check_create_mask(self, dpi = 80, figsize = (0.7*20,0.7*5), fontsize = 10):
        fig, ax = plt.subplots(1,4, figsize = figsize, constrained_layout=True, dpi = dpi )

        x,y = zip(*self.pos_spots_img_idx)
        
        ax[0].imshow(self.mask_untransformed/self.img_mask.mean()*self.mask_untransformed.mean(), cmap = plt.cm.gray_r, zorder = 1, alpha = 0.3)
        ax[0].imshow(self.img_mask, cmap = plt.cm.Reds, zorder = 0)
        ax[0].set_title('identified masked area', fontsize = fontsize)
        ax[0].axis('off')
         
        ax[1].imshow(np.log(self.img_spots), interpolation='nearest', cmap = plt.cm.gray_r)
        ax[1].scatter(x,y, s=80, facecolors='none', edgecolors='r', label = 'identified spots')
        ax[1].set_title('identified clusters', fontsize = fontsize)
        ax[1].axis('off')        
        
        x_coord_img, y_coord_img = zip(*self.pos_spots_img_idx_sorted)
        x_coord_holo, y_coord_holo = zip(*self.pos_spots_holo_idx_sorted)
        
        legend_elements = [plt.Line2D([0], [0], markerfacecolor='b',  color='w', marker = '*', label='transformed sideport positions', markersize = 20),
                           plt.Line2D([0], [0], marker='o',  color='w', markerfacecolor='crimson', alpha = 0.5, label='trap positions holo', markersize = 12)]
        
        [ax[2].scatter(*zip(*self.t_warping(val)), c = 'black', marker = '*', s=80, label = 'transformed coord') for val in np.array(self.pos_spots_holo_idx)]
        ax[2].scatter(x_coord_img, y_coord_img, c = 'crimson', marker = 'o', s=100, alpha = 0.5)
        ax[2].grid(True)
        ax[2].legend(handles=legend_elements, fontsize = fontsize*0.8)#, loc='center')
        ax[2].set_title('transformed coordinates', fontsize = fontsize)
        
        ax[3].imshow(self.mask.T, cmap = plt.cm.Blues, clim = [0,1])
        ax[3].axis('off')
        ax[3].set_title('Mask', fontsize = fontsize)
        
if __name__ == '__main__': 
    pass