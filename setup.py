from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy 
import sys

if sys.platform == "win32":
    extra_compile_args = ["/std:c++17"]
else:
    extra_compile_args = ["-std=c++17"] 
    

ext = Extension(
     name="regtree._core",
     sources=[
         "regtree/_core.pyx", 
         "regtree/RegressionTree.cpp"
     ],
     include_dirs=["regtree", numpy.get_include()],
     language="c++",
     extra_compile_args=extra_compile_args  
 )

setup(
    name="regtree",
    version="0.1.0",
    ext_modules=cythonize(ext, compiler_directives={"language_level": "3"}),
    packages=["regtree"],
    install_requires=[
        "numpy",
        "Cython"
    ],

)