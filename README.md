[<img align="right" src="https://user-images.githubusercontent.com/106104347/186722156-163baeb0-108d-4a3b-b962-24141d868dd8.png" align="center" width="250"/>](https://user-images.githubusercontent.com/106104347/186722156-163baeb0-108d-4a3b-b962-24141d868dd8.png#gh-dark-mode-only)

[<img align="right" src="https://user-images.githubusercontent.com/106104347/186722204-7605eefa-68ab-4546-bd4b-4c7b9cc14c4d.png" width="250"/>](https://user-images.githubusercontent.com/106104347/186722204-7605eefa-68ab-4546-bd4b-4c7b9cc14c4d.png#gh-light-mode-only)

![license](https://img.shields.io/github/license/rafaela-felix/pyddt)
![size](https://img.shields.io/github/languages/code-size/rafaela-felix/pyddt)
![language](https://img.shields.io/github/languages/top/rafaela-felix/pyddt)

**pyddt** is an open-source Python package for exploiting dynamic X-ray diffraction in single crystals. It allows experiment planning aimed at structure refinement of known materials, complete analysis of experimental data set, and elaboration of improved model structure compatible with phase triplet information extracted by the incorporated data analysis tools. 

**pyddt** features include:

* Structural modelling from CIF.
* Easy probing of the dynamical-diffraction susceptibility to structural details such as valence of chemical species, vacancies, anti-site occupation, internal strain due to foreign atoms, and relative differences in atomic displacement.
* Identification of optimum resonant diffraction conditions.
* Extracting all structure factor phase information available in the data set.
* Comparison of experimental and theoretical phase triplets.

# Contents

* pyddt: directory with the source codes for the Python package.
* docs: directory with documentation (html and pdf versions).
  * docs/notebooks: directory with notebook tutorials.
* setup.py: *setuptools* install script for the package installation.

# Installation

First, use 

```
git clone https://github.com/rafaela-felix/pyddt.git
```

for obtaining the source code. Then, installing pyddt by executing

```
pip install .
```

inside the source folder on the command line.

## Python package configuration

Considering that pyddt package has been installed in a directory unknown to your Python distribution, you have to indicate Python where to look for it. So, you might:

* add the installation directory to your PYTHONPATH environment variable.
* add the path to *sys.path* in the *.pythonrc* file placed in your home directory.
* add the path to *sys.path* in every script where you want to use the package before importing it.

```
import sys
sys.path.append("path to the pyddt package")

import pyddt
```
It might need to use a double backslash `\\` in the file path on Windows systems.

## Testing 

For now, pyddt does not contain formal tests. 

If you can run the tutorial notebooks (*docs/notebooks/* directory), you have successfully installed the package.


# Requirements

The following requirements are needed for installing and using pyddt:

* Python (version >= 3.9)
* Python packages (pip installable):
  * numpy (>= 1.22.0)
  * matplotlib (>= 3.5.0)
  * scipy (>= 1.8.0)
  * pandas (>= 1.4.1)
  * plotly (>= 5.6.0)
  * sklearn
  * lmfit (>= 1.0.3)
  * ase (>= 3.22.1)
  * tqdm (>= 4.63.1)
  * pymatgen (>= 2022.5.26)
  * nglwview (only used in Jupyter Notebooks, version >= 3.0.3)
  
 **Windows systems**: The current version of *pymatgen* is not correctly installed using `pip`. In this case, please use `conda`.
 
 ```
 conda install --channel conda-forge pymatgen
 ```
  
 # Getting started

 To get started with pyddt, see our [user guide and tutorials](https://rafaela-felix.github.io/pyddt/tutorial.html).
   
 # Documentation
 
 For more details on the use of pyddt, check out the *docs* directory or our [online documentation](https://rafaela-felix.github.io/pyddt/api.html).
 
# Acknowledgements

This package was developed at the University of São Paulo with FAPESP funding (grant ns. 2021/01004-6 and 2019/019461-1).

# Citation

If you use pyddt in your research, please consider citing the following work:

```
@article{pyddt,
  title={},
  author={},
  journal={},
  pages={},
  year={2022},
  doi={}
  publisher={}
}
```

