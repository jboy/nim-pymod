/*
 * Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
 * All rights reserved.
 *
 * This source code is licensed under the terms of the MIT license
 * found in the "LICENSE" file in the root directory of this source tree.
 */

#ifndef PYMODPYUTILS_C_H
#define PYMODPYUTILS_C_H

#include <Python.h>

size_t
getPyRefCnt(PyObject *obj);

PyObject *
raisePyAssertionError(const char *msg);

PyObject *
raisePyIndexError(const char *msg);

PyObject *
raisePyKeyError(const char *msg);

PyObject *
raisePyRuntimeError(const char *msg);

PyObject *
raisePyTypeError(const char *msg);

PyObject *
raisePyValueError(const char *msg);

PyObject *
getPyNone();

#endif  /* PYMODPYUTILS_C_H */
