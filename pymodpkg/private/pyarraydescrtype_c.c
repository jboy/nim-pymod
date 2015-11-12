/*
 * Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
 * All rights reserved.
 *
 * This source code is licensed under the terms of the MIT license
 * found in the "LICENSE" file in the root directory of this source tree.
 */

#include "pyarraydescrtype_c.h"

char
PyArrayDescr_kind(const PyArray_Descr *d) {
    return d->kind;
}

char
PyArrayDescr_type(const PyArray_Descr *d) {
    return d->type;
}

char
PyArrayDescr_byteorder(const PyArray_Descr *d) {
    return d->byteorder;
}

/*
 * NOTE:  In the Numpy docs, the `flags` attribute of the PyArray_Descr struct
 * is described as being of int type (and following an attribute of char type
 * called `unused`).  However, in the C header file "<numpy/ndarraytypes.h>",
 * the `flags` attribute is defined with char type (and there is no `unused`
 * attribute).
 */
char
PyArrayDescr_flags(const PyArray_Descr *d) {
    return d->flags;
}

int
PyArrayDescr_type_num(const PyArray_Descr *d) {
    return d->type_num;
}

int
PyArrayDescr_elsize(const PyArray_Descr *d) {
    return d->elsize;
}

int
PyArrayDescr_alignment(const PyArray_Descr *d) {
    return d->alignment;
}
