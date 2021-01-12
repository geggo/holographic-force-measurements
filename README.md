# holoforce: data analysis software for direct measurement of individual optical forces

This repository contains the source code of programs for holographic force measurements, as described in:
> [Franziska Strasser, Simon Moser, Monika Ritsch-Marte, and Gregor Thalhammer. Direct measurement of individual optical forces in ensembles of trapped particles. Optica 8(1), 79-87 (2021)](https://www.osapublishing.org/optica/fulltext.cfm?uri=optica-8-1-79&id=446489O)

Additional code used for experiment control is contained in the [tweezers](tweezers) directory.

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

