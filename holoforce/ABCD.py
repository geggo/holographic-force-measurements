# -*- coding: utf-8 -*-

"""
Tools for handling ABCD matrices in optics
"""

import numpy as np


def concat(*elements):
    """concatenate ABCD arrays, given from object to image"""
    m = np.eye(2)
    for elem in elements[::-1]:
        m = m @ elem
    return m


def prop(d):
    """free space propagation"""
    return np.array([[1., d], [0., 1.]])


def lens(f):
    """lens with focal length f"""
    return np.array([[1., 0.], [-1./f, 1.]])


def to_inf(d1=0.3, d2=0.3, f=0.3):
    """lens with separations d1 and d2 from object and image plane"""
    return concat(prop(d1), lens(f), prop(d2))


def lens_at_f(f=300e-3):
    """Fourier lens, with """
    return to_inf(f=f, d1=f, d2=f)
