import setuptools

setuptools.setup(
    name='util',
    instlal_requires=[
        "numpy",
        "matplotlib",
        "pytest",
        "pandas",
	    "tqdm",
        "pynvim",
        "black",
        "pylint",
        "adjustText",
        "pypdf",
        "scipy",
        "ase==3.23",
        "wfl",

    ],
    packages=["util"],
)
