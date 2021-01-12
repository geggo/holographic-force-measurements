# tweezers

This directory contains part of the experiment control software. 

## Requirements

Software is implemented in Python. Some additional packages are needed

data processing

* numpy
* scipy
* scikit-image

GPU acceleration (using OpenCL)

* pyopencl

fast graphic rendering (using OpenGL)

* PyOpenGL

graphical user interface:

* traitsui
* qt (PyQt or pyside2)

## Calibration and compensation of optical distortion

Run demonstration program:

	python CalibratorBFP.py
	
