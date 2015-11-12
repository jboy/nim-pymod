# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

#
#=== Some useful PyObject types to get you started
#

definePyObjectType(PyObject, "PyObject",
    "", "Any", ["<Python.h>"])

definePyObjectType(PyArrayObject, "PyArrayObject",
    "PyArray_Type", "numpy.ndarray", ["pymodpkg/private/numpyarrayobject.h"])

