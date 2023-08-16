import numpy, scipy
import os, sys
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import Cython.Compiler.Options
Cython.Compiler.Options.annotate=True

extra_compile_args = ["-Ofast"]
#extra_compile_args = ["-O3"]

ext_modules = [
    Extension("modemc/*", ["modemc/*.pyx"], include_dirs=['.', numpy.get_include()], extra_compile_args=extra_compile_args),
    Extension("modemc.protocols/*", ["modemc/protocols/*.pyx"], include_dirs=['.', numpy.get_include()], extra_compile_args=extra_compile_args),
    Extension("modemc.systems/*", ["modemc/systems/*.pyx"], include_dirs=['.', numpy.get_include()], extra_compile_args=extra_compile_args)
]

setup(
    name='modemc',
    version='1.0.0',
    url='https://github.com/lukastk/',
    author='Lukas Kikuchi',
    license='MIT',
    description='',
    long_description='',
    platforms='works on all platforms (such as LINUX, macOS, and Microsoft Windows)',
    ext_modules=cythonize(ext_modules,
        compiler_directives={"language_level": sys.version_info[0]},
        ),
    libraries=[],
    packages=[
        'modemc',
        'modemc.protocols',
        'modemc.systems'
    ],
    package_data={
        'modemc': ['*.pxd'],
        'modemc.protocols': ['*.pxd'],
        'modemc.systems': ['*.pxd'],
    },
)
