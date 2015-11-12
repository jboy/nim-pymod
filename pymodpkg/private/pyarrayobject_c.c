/*
 * Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
 * All rights reserved.
 *
 * This source code is licensed under the terms of the MIT license
 * found in the "LICENSE" file in the root directory of this source tree.
 */

#include "pyarrayobject_c.h"


PyArray_Descr *
getDescrFromType(int typenum) {
    PyArray_Descr *res = PyArray_DescrFromType(typenum);
    return res;
}


int
canCastArrayToImpl(PyArrayObject *arr, PyArray_Descr *totype, int casting) {
    int res = PyArray_CanCastArrayTo(arr, totype, casting);
    return res;
}


PyArrayObject *
createNewLikeArrayImpl(PyArrayObject *prototype, int order, PyArray_Descr *dtype,
        int subok) {
    PyArrayObject *res = PyArray_NewLikeArray(prototype, order, dtype, subok);
    return res;
}


PyArrayObject *
createSimpleNewImpl(int nd, npy_intp *dims, int typenum) {
    /*
     * Fun fact:  `PyArray_SimpleNew` is actually a C preprocessor macro,
     * even though the Numpy docs describe it describe it like a function:
     *  http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_SimpleNew
     */
    PyArrayObject *res = PyArray_SimpleNew(nd, dims, typenum);
    return res;
}


PyArrayObject *
createNewCopyNewDataImpl(PyArrayObject *old, int order) {
    PyArrayObject *res = PyArray_NewCopy(old, order);
    return res;
}


int
doCopyIntoImpl(PyArrayObject *dest, PyArrayObject *src) {
    int res = PyArray_CopyInto(dest, src);
    return res;
}


void
doFILLWBYTE(PyArrayObject *arr, int val) {
    PyObject *p = (PyObject *)arr;
    /*
     * It's a C preprocessor macro:
     *  http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_FILLWBYTE
     */
    PyArray_FILLWBYTE(p, val);
}


void
doResizeDataInplaceImpl(PyArrayObject *old, int nd, npy_intp *dims, int refcheck) {
    PyArray_Dims pa_dims;
    pa_dims.ptr = dims;
    pa_dims.len = nd;
    /*
     * Note that we supply a hard-coded value of `NPY_ANYORDER` as the 4th
     * argument to `PyArray_Resize`.  As the docs for `PyArray_Resize` explain:
     *
     *   "The fortran argument can be NPY_ANYORDER, NPY_CORDER, or NPY_FORTRANORDER.
     *   It currently has no effect.  Eventually it could be used to determine how
     *   the resize operation should view the data when constructing a
     *   differently-dimensioned array."
     */
    PyArray_Resize(old, &pa_dims, refcheck, NPY_ANYORDER);
}
