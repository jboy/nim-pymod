/*
 * Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
 * All rights reserved.
 *
 * This source code is licensed under the terms of the MIT license
 * found in the "LICENSE" file in the root directory of this source tree.
 */

#ifndef PYARRAYDESCR_C_H
#define PYARRAYDESCR_C_H

#include "numpyarrayobject.h"

/*
 * Provide (read-only) accessors for the attributes of struct PyArray_Descr:
 *  http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArray_Descr
 */

char
PyArrayDescr_kind(const PyArray_Descr *d);

char
PyArrayDescr_type(const PyArray_Descr *d);

char
PyArrayDescr_byteorder(const PyArray_Descr *d);

/*
 * NOTE:  In the Numpy docs, the `flags` attribute of the PyArray_Descr struct
 * is described as being of int type (and following an attribute of char type
 * called `unused`).  However, in the C header file "<numpy/ndarraytypes.h>",
 * the `flags` attribute is defined with char type (and there is no `unused`
 * attribute).
 */
char
PyArrayDescr_flags(const PyArray_Descr *d);

int
PyArrayDescr_type_num(const PyArray_Descr *d);

int
PyArrayDescr_elsize(const PyArray_Descr *d);

int
PyArrayDescr_alignment(const PyArray_Descr *d);

#endif  /* PYARRAYDESCR_C_H */
