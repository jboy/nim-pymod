/*
 * Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
 * All rights reserved.
 *
 * This source code is licensed under the terms of the MIT license
 * found in the "LICENSE" file in the root directory of this source tree.
 */

#ifndef NUMPYARRAYOBJECT_H
#define NUMPYARRAYOBJECT_H

/*
 * It seems that <numpy/arrayobject.h> depends upon the contents of <Python.h>,
 * but doesn't actually #include <Python.h> into itself, instead relying upon
 * the includer of <numpy/arrayobject.h> to also include <Python.h> beforehand
 * in the same C file.
 *
 * Unfortunately, this is not always something under the programmer's control.
 *
 * For example, Nim's `header` pragma can only be used once per Nim object
 * type definition (when you are importing a struct definition from C using
 * the `importc` pragma), and it can only accept a string as an argument,
 * and that string can only contain a single header file name.
 *
 * Hence, we need to create this single header file, that includes <Python.h>
 * prior to including <numpy/arrayobject.h>.  Lame...
 */
#include <Python.h>

/*
 * http://docs.scipy.org/doc/numpy/reference/c-api.deprecations.html#deprecation-mechanism-npy-no-deprecated-api
 */
#define NPY_NO_DEPRECATED_API  NPY_1_7_API_VERSION

/*
 * Ensure that we can access the static Numpy data, which is initialised in
 * the main Python C-API file by Numpy's `import_array()` macro, from other
 * C files.  If we don't to this, each independent C file will define its
 * own static Numpy data in its own compilation unit, but not initialise it
 * (because `import_array()` wasn't invoked for that compilation unit), which
 * will result in any Numpy API invocations happily using uninitialised data.
 *
 * Yeah, that was a fun one to debug.
 *
 * More information:
 *  http://stackoverflow.com/a/12259667
 *  http://docs.scipy.org/doc/numpy/reference/c-api.array.html#importing-the-api
 */
#ifndef YES_IMPORT_ARRAY
#define NO_IMPORT_ARRAY
#endif
#define PY_ARRAY_UNIQUE_SYMBOL pymod_ARRAY_API

/*
 * And finally, the header you've all been waiting for!...
 */
#include <numpy/arrayobject.h>

#endif  /* NUMPYARRAYOBJECT_H */
