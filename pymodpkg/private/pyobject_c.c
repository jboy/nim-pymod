/*
 * Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
 * All rights reserved.
 *
 * This source code is licensed under the terms of the MIT license
 * found in the "LICENSE" file in the root directory of this source tree.
 */

#include <stdio.h>
#include <stdlib.h>

#include "pyobject_c.h"


size_t
getPyRefCnt(PyObject *obj) {
	return Py_REFCNT(obj);
}


/*
 * NOTE: In the `raise_XError` functions below, the string message `msg` is
 * allocated by Nim, and will presumablye de-allocated by Nim at some point
 * in the future.  Hence, we ask: Will Python make its own copy of the string?
 *
 * The answer appears to be "Yes".
 *
 * From "Python/errors.c":
 *   PyObject
 *   PyErr_SetString(PyObject *exception, const char *string)
 *   {
 *       PyObject *value = PyString_FromString(string);
 *       PyErr_SetObject(exception, value);
 *       Py_XDECREF(value);
 *   }
 *
 * From "Objects/stringobject.c":
 *  PyObject *
 *  PyString_FromString(const char *str)
 *  {
 *      register size_t size;
 *      register PyStringObject *op;
 *  
 *      assert(str != NULL);
 *      size = strlen(str);
 * [...]
 *      op = (PyStringObject *)PyObject_MALLOC(PyStringObject_SIZE + size);
 *      if (op == NULL)
 *          return PyErr_NoMemory();
 *      PyObject_INIT_VAR(op, &PyString_Type, size);
 *      op->ob_shash = -1;
 *      op->ob_sstate = SSTATE_NOT_INTERNED;
 *      Py_MEMCPY(op->ob_sval, str, size+1);
 * [...]
 *      return (PyObject *) op;
 *  }
 */

PyObject *
raisePyAssertionError(const char *msg) {
	PyErr_SetString(PyExc_AssertionError, msg);
	return NULL;
}


PyObject *
raisePyIndexError(const char *msg) {
	PyErr_SetString(PyExc_IndexError, msg);
	return NULL;
}


PyObject *
raisePyKeyError(const char *msg) {
	PyErr_SetString(PyExc_KeyError, msg);
	return NULL;
}


PyObject *
raisePyRuntimeError(const char *msg) {
	PyErr_SetString(PyExc_RuntimeError, msg);
	return NULL;
}


PyObject *
raisePyTypeError(const char *msg) {
	PyErr_SetString(PyExc_TypeError, msg);
	return NULL;
}


PyObject *
raisePyValueError(const char *msg) {
	PyErr_SetString(PyExc_ValueError, msg);
	return NULL;
}


PyObject *
getPyNone() {
	Py_INCREF(Py_None);
	return Py_None;
}

