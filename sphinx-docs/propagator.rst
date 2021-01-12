Propagation of light fields
===========================

A central step is to numerically propagate light fields between planes, possibly with some optics in between.

The class
:class:`~holoforce.propagator.FresnelPropagator` and its GPU acceleratated sibling :class:`~holoforce.propagator.FresnelPropagatorGPU` provide a implementation based on the Fresnel diffraction integral, generalized (Collins integral) to propagation through optics that is described by ABCD-matrices. The implementation imposes no restrictions on the spatial sampling of the light field.

The :mod:`~holoforce.ABCD` module provides some convenience functions to create the ABCD matrices, e.g., for free space propagation and passing through a thin lens (in paraxial approximation).

The parameters describing the optics (wavelength, ABCD matrix) are bundled in a :class:`~holoforce.propagator.ParaxialOptics` object.

Tools for creating ABCD matrices to describe propagation through optics
-----------------------------------------------------------------------

.. automodule:: holoforce.ABCD
   :members: prop, lens, lens_at_f

	     
Propagators
-----------

.. automodule:: holoforce.propagator
   :members: FresnelPropagator, FresnelPropagatorGPU, ParaxialOptics
   :show-inheritance:
   :member-order: bysource	     
