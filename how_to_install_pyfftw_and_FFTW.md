1. Download FFTW https://www.fftw.org/download.html
    1. Run the following commands:
        ./configure --enable-shared --enable-threads --enable-openmp --prefix /store/SOFT/ltk26/install/
        make CFLAGS=-fPIC
        make install
        
1. Clone the pyfftw repository https://github.com/pyFFTW/pyFFTW
    1. Edit setup.py so that the include and lib folders in /store/SOFT/ltk26/install are found. Alternatively, copy over 'pyfftw_setup.py' which has the changes.
    2. python setup.py install

Various links that helped me:

- https://github.com/pyFFTW/pyFFTW/issues/294
- https://stackoverflow.com/questions/19768267/relocation-r-x86-64-32s-against-linking-error
- https://stackoverflow.com/questions/45321342/how-to-build-fftw-in-ubuntu