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
    ],
    packages=["util"],
)
