# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

## Several short examples of how to create a PyArrayObject instance
## that can be returned to Python.
##
## Compile this Nim module using the following command:
##   python ../../pmgen.py arraycreation.nim

import strutils  # `%`
import pymod
import pymodpkg/docstrings
import pymodpkg/pyarrayobject


proc arange_int32*(num: int): ptr PyArrayObject {.exportpy} =
  docstring"""Create a new Numpy-like `arange` of `num` elements of dtype `int32`.

  The elements will start at 0 and increase monotonically to (num - 1).

  This example uses the `createSimpleNew` wrapper proc provided by Pymod.
  """
  var num = num  # Make `num` mutable in Nim.
  if num < 0:
    # Mimic the default behaviour of `numpy.arange`: round `num` up to 0.
    num = 0

  result = createSimpleNew([num], np_int32)
  var i: int32 = 0  # Note: `int32` is not the same type as `int`.
  for mval in result.iterateFlat(int32).mitems:
    mval = i
    inc(i)


proc empty_like*(arr: ptr PyArrayObject): ptr PyArrayObject {.exportpy.} =
  docstring"""Create a new, uninitialized Numpy array with shape & dtype like `arr`.

  This example uses the `createSimpleNew` wrapper proc provided by Pymod.

  A more flexible approach is to use the `createNewLikeArray` provided by Pymod
  (which is a wrapper around the Numpy C-API function `PyArray_NewLikeArray`).
  """
  result = createSimpleNew(arr.shape, arr.dtype)


proc zeros_like*(arr: ptr PyArrayObject): ptr PyArrayObject {.exportpy.} =
  docstring"""Create a new, zero-initialized Numpy array with shape & dtype like `arr`.

  This example uses the `createSimpleNew` & `doFILLWBYTE` wrapper procs provided
  by Pymod.

  A more flexible approach is to use the `createNewLikeArray` provided by Pymod
  (which is a wrapper around the Numpy C-API function `PyArray_NewLikeArray`).
  """
  result = createSimpleNew(arr.shape, arr.dtype)
  doFILLWBYTE(result, 0)


proc createRgbImage*(numRows, numColumns: int; initVal: cint=0):
    ptr PyArrayObject {.exportpy} =
  docstring"""Create a 3-D array of `uint8` to represent a 3-channel RGB image.

  Specify the number of rows & columns in the image.

  The pixels will be initialized to the supplied `cint` value `initVal`.
  If the value is less than 0, it will be rounded up to 0; if the value is
  greater than 255, it will be rounded down to 255.

  This example uses the `createSimpleNew` wrapper proc provided by Pymod.
  """
  if numRows <= 0:
    raise newException(ValueError, "number of rows in image must be > 0")
  if numColumns <= 0:
    raise newException(ValueError, "number of columns in image must be > 0")

  result = createSimpleNew([numRows, numColumns, 3], np_uint8)

  var initVal = initVal  # Make `initVal` mutable in Nim.
  if initVal < 0:
    initVal = 0
  elif initVal > 255:
    initVal = 255
  doFILLWBYTE(result, initVal)


initPyModule("_arraycreation", arange_int32, empty_like, zeros_like, createRgbImage)
