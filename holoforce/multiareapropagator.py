# -*- coding: utf-8 -*-

import attr
import pyopencl as cl
import numpy as np

import ABCD
from propagator import Field, ParaxialOptics, FresnelPropagatorGPU
from cltools import CLConsumer


@attr.s
class MultiAreaField(object):
    """
    field with subfields of arbitrary shape
    """
    shape = attr.ib(default=(1024, 1024), repr=False)
    size = attr.ib(default=(10e-3, 10e-3), repr=False)
    center = attr.ib(default=(0., 0.), repr=False)

    field : Field = attr.ib()  #: Field : base field
    @field.default
    def _field_default(self):
        return Field(shape=self.shape,
                     size=self.size,
                     center=self.center)

    # TODO: if field given externally, update shape, size, center, ...

    subfields : list = attr.ib(factory=list)  #: list[Field] : list of subfields, i.e., slices into  :py:attr:`.field`
    subfields_masks = attr.ib(factory=list)  #: list[boolean array] : list of masks for :py:attr:`.subfields`

    # TODO make subfields combined an independent array, not a view
    subfields_combined : Field = attr.ib(init=False)  #: Field : all subfields combined, slice into :attr:`.field`
    masks_subfields_combined : list = attr.ib(init=False)  #: list[array] : list of masks for each subfield, same size as :attr:`.subfields_combined`
    mask_subfields_combined : np.ndarray = attr.ib(init=False)  #: all `.masks_subfields_combined` or'd together

    def append_subfield(self, sub_field: Field, mask=None):
        """
        append view to sub region to list of subfields

        Parameters
        ----------
        sub_field : array
            needs to be view (slice) to sub region of self.field
        mask : boolean array, optional
            same shape as `sub_field`
        """
        assert sub_field.field.base is self.field.field
        self.subfields.append(sub_field)
        self.subfields_masks.append(mask)

    def combine_subfields(self):
        """
        combine subfields to larger subfield containing all, stored in :attr:`.subfields_combined`

        for each subfield create mask with size of combined subfields, and stores it in :attr:`.masks_subfields_combined`
        """

        cc = []
        for subfield in self.subfields:
            # coords of subfield rectangle
            (x_min, x_max), (y_min, y_max) = [(c[0], c[-1]) for c in subfield.x]
            i0, k0 = self.field.find_idx_from_pos((x_min, y_min))  # indices of fourier_field
            i1, k1 = self.field.find_idx_from_pos((x_max, y_max))
            cc.append([i0, k0])
            cc.append([i1, k1])

        # find indices of rectangle enclosing all subfields
        i0, k0 = np.array(cc).min(axis=0)
        i1, k1 = np.array(cc).max(axis=0)
        iw, kw = roundup8(i1-i0+1), roundup8(k1-k0+1) # increase size to multiple of 8
        if i0 + iw >= self.field.shape[0]:
            i0 -= (i0+iw+1) - self.field.shape[0]
        if k0 + kw >= self.field.shape[1]:
            k0 -= (k0+kw+1) - self.field.shape[1]

        self.subfields_combined = self.field[i0:i0 + iw, k0:k0 + kw]

        # create subfields_masks for combined subfields
        self.masks_subfields_combined = []
        for subfield, mask in zip(self.subfields, self.subfields_masks):  # TODO: only for children of subfields_combined
            (x_min, x_max), (y_min, y_max) = [(c[0], c[-1]) for c in subfield.x]
            sx_min, sy_min = self.subfields_combined.find_idx_from_pos((x_min, y_min))
            sx_max, sy_max = self.subfields_combined.find_idx_from_pos((x_max, y_max))

            mask_combined = np.zeros_like(self.subfields_combined.field, dtype=np.bool)
            mask_combined[sx_min:sx_max+1, sy_min:sy_max+1] = mask if mask is not None else True

            self.masks_subfields_combined.append(mask_combined)

        # combine individual masks
        masks = self.masks_subfields_combined
        mask_combined = masks[0]
        for mask in masks[1:]:
            mask_combined = mask_combined | mask
        self.mask_subfields_combined = mask_combined

def roundup8(x: int) -> int:
    """
    round up argument to nearest multiple of 8

    Parameters
    ----------
    x : int
        argument

    Returns
    -------
    int
        `x` rounded up to next multiple of 8
    """
    return ((x - 1) | 8 - 1) + 1




@attr.s
class MultiAreaFocalToFarfieldPropagatorGPU(CLConsumer):
    """
    Propagator for multi area field to farfield
    """
    
    optics: ParaxialOptics = attr.ib(kw_only=True)

    @optics.default
    def _optics_default(self):
        return ParaxialOptics(ABCD=ABCD.to_inf())

    #: multi area focal field
    focal : MultiAreaField = attr.ib(kw_only = True,
                                     validator=attr.validators.optional(attr.validators.instance_of(MultiAreaField))
                                     )

    #: far field
    pupil : Field = attr.ib(kw_only = True)

    #: propagator for focal.subfields_combined to pupil
    propagator_combined : FresnelPropagatorGPU = attr.ib(kw_only = True)
    @propagator_combined.default
    def _propagator_combined_default(self):
        return FresnelPropagatorGPU(self.focal.subfields_combined, self.pupil,
                                    paraxial_optics=self.optics,
                                    cl_queue=self.cl_queue)

    #: propagator for full field (focal.field) to pupil
    propagator_full_field = attr.ib(kw_only=True)
    @propagator_full_field.default
    def _propagator_full_field_default(self):
        return FresnelPropagatorGPU(self.focal.field, self.pupil,
                                    paraxial_optics=self.optics,
                                    cl_queue=self.cl_queue)


if __name__ == '__main__':
    pass
    '''
    cl_platform = cl.get_platforms()[0]
    cl_device = cl_platform.get_devices(cl.device_type.GPU)[0]
    cl_context = cl.Context([cl_device])

    mafocal = MultiAreaField(shape=(400, 160), size=(4, 2))
    subfield0 = mafocal.field[50:150, 50:]
    mask0 = ((subfield0.X[0] - subfield0.center[0])**2 + (subfield0.X[1] - subfield0.center[1])**2) <  (subfield0.size[0]**2 * 0.25)

    mafocal.append_subfield(subfield0, mask=mask0)
    mafocal.append_subfield(mafocal.field[200:250, 100:])
    mafocal.subfields[0].field[::2] = 1.
    mafocal.subfields[1].field[:, ::2] = 1.

    mafocal.combine_subfields()

    pupil = Field(shape=(1000, 1000), size=(20e-3, 20e-3))

    maprop = MultiAreaFocalToFarfieldPropagatorGPU(focal = mafocal,
                                                   pupil = pupil,
                                                   cl_context=cl_context)

    maprop.focal = mafocal
    '''