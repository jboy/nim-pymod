{. compile: "pymodpkg/private/pyobject_c.c" .}

# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

type PyObject* {. importc: "PyObject", header: "<Python.h>", final .} = object

# https://docs.python.org/2/c-api/arg.html#c.Py_BuildValue
# http://nim-lang.org/manual.html#varargs-pragma
proc Py_BuildValue*(fmt: cstring): ptr PyObject {.
  importc: "Py_BuildValue", header: "<Python.h>", cdecl, varargs .}

proc getPyRefCnt*(obj: ptr PyObject): csize {.
  importc: "getPyRefCnt", header: "pymodpkg/private/pyobject_c.h" .}

proc doPyIncRef*(obj: ptr PyObject): void {.
  importc: "Py_IncRef", header: "<Python.h>" .}

proc doPyDecRef*(obj: ptr PyObject): void {.
  importc: "Py_DecRef", header: "<Python.h>" .}

proc raisePyAssertionError*(msg: cstring): ptr PyObject {.
  importc: "raisePyAssertionError", header: "pymodpkg/private/pyobject_c.h" .}

proc raisePyIndexError*(msg: cstring): ptr PyObject {.
  importc: "raisePyIndexError", header: "pymodpkg/private/pyobject_c.h" .}

proc raisePyKeyError*(msg: cstring): ptr PyObject {.
  importc: "raisePyKeyError", header: "pymodpkg/private/pyobject_c.h" .}

proc raisePyRuntimeError*(msg: cstring): ptr PyObject {.
  importc: "raisePyRuntimeError", header: "pymodpkg/private/pyobject_c.h" .}

proc raisePyTypeError*(msg: cstring): ptr PyObject {.
  importc: "raisePyTypeError", header: "pymodpkg/private/pyobject_c.h" .}

proc raisePyValueError*(msg: cstring): ptr PyObject {.
  importc: "raisePyValueError", header: "pymodpkg/private/pyobject_c.h" .}

proc getPyNone*(): ptr PyObject {.
  importc: "getPyNone", header: "pymodpkg/private/pyobject_c.h" .}

