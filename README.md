# Direct measurement of individual optical forces: data analysis software

This repository contains the source code for holographic force measurements.

**NOTE** This is work in progress

## Installation

The programs are implemented using Python. To install use

    python setup.py install

For developing the software we recommend to install the package in development mode

    python setup.py develop


## Usage

Usage of the software is demonstrated in IPython notebooks contained in the [examples](examples) directory.

## Documentation

To create documentation in HTML format (output placed at `build/html`) use
	
	python setup.py build_sphinx 
	
or using make

	make html
