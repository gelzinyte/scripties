import setuptools

setuptools.setup(
    name='util',
    instlal_requires=[
        "numpy",
        "matplotlib",
        "pytest",
        "pandas",
        "ipython",
        "pynvim",
	    "tqdm",
        "pynvim",
        "black",
        "pylint",
        "adjustText",
        "pypdf",
        "scipy",
        "ase==3.23",
        "wfl",
        "jedi",
    ],
    packages=["util"],
)
