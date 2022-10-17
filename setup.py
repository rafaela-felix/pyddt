import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyddt",
    version="0.0.2",
    author="Rafaela Felix",
    author_email="rafaelafelixp@usp.br",
    description="Tool for planning and analyzing XRDD experiments.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rafaela-felix/pyddt/",
    keywords="x-ray diffraction dynamic-diffraction renninger-scan",
    license="GNU General Public License (GPL)",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "pyddt"},
    packages=setuptools.find_packages(where="pyddt"),
    install_requires=[
        'matplotlib>=3.5.1',
        'numpy>=1.22.4',
        'pandas>=1.4.1', 
        'plotly>=5.6.0',
        'scipy>=1.8.0',
        'tqdm>=4.63.1',
        'lmfit>=1.0.3',
        'sklearn',
        'ase>=3.22.1',
        'nglview>=3.0.3',
        'pymatgen>=2022.5.26'
    ],
    python_requires=">=3.6",
)

