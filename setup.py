from distutils.core import setup, Extension
from Cython.Build import cythonize
import eigency
import numpy

extensions = [
    Extension(
        "propagation",
        ["propagation.pyx"],
        language="c++",
        extra_compile_args=["-std=c++11", "-lpthread", "-O3"],
        include_dirs=[".", numpy.get_include()] + eigency.get_includes(),
    )
]

setup(
    name="propagation",
    author="LuYi",
    version="0.0.1",
    ext_modules=cythonize(extensions),
    install_requires=["Cython>=0.2.15", "eigency>=1.77"],
    packages=["little-try"],
    python_requires=">=3",
)
