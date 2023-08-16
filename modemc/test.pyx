import numpy as np
cimport numpy as np
cimport cpython

cdef extern from "math.h":
    double exp(double x)

ctypedef int (*func_foo)(double[:], int)

cdef int test(double[:] a, int b):
    print(b)
    print(np.array(a))
    return 2

cdef test2(func_foo f):
    cdef double c = np.exp(1231223)
    cdef double e = exp(7124123)

    a = np.array([1,2,3.0])
    b = 2
    f(a, b)

cpdef test3():
    test2(test)


cdef class X():
    cdef func_foo test

    def __init__(self):
        self.test = test

cdef class Y():
    cdef func_foo foo

    def __init__(self, X x):
        self.foo = x.test

    def test2(self):
        a = np.array([1,2,3.0])
        b = 2
        self.foo(a, b)

t1 = np.array([1,2,3.0])
cdef double[:] t2 = t1
t3 = np.array(t2)
t3[0] = 5123
print(t1)


cdef hello(double[:] hey):
    print(np.array(hey))
cdef double[:,:] mm = np.zeros( (4,4) )
hello(mm[0,:])