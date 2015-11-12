/*
 * Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
 * All rights reserved.
 *
 * This source code is licensed under the terms of the MIT license
 * found in the "LICENSE" file in the root directory of this source tree.
 */

#ifndef PYARRAYOBJECT_C_H
#define PYARRAYOBJECT_C_H

#include "numpyarrayobject.h"

/* We define these functions when Numpy only provides C preprocessor macros. */

PyArray_Descr *
getDescrFromType(int typenum);

int
canCastArrayToImpl(PyArrayObject *arr, PyArray_Descr *totype, int casting);

PyArrayObject *
createNewLikeArrayImpl(PyArrayObject *prototype, int order, PyArray_Descr *dtype,
        int subok);

PyArrayObject *
createSimpleNewImpl(int nd, npy_intp *dims, int typenum);

PyArrayObject *
createNewCopyNewDataImpl(PyArrayObject *old, int order);

int
doCopyIntoImpl(PyArrayObject *dest, PyArrayObject *src);

void
doFILLWBYTE(PyArrayObject *arr, int val);

void
doResizeDataInplaceImpl(PyArrayObject *old, int nd, npy_intp *dims, int refcheck);

#endif  /* PYARRAYOBJECT_C_H */
