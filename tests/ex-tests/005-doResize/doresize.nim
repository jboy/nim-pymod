# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import strutils

import pymod
import pymodpkg/docstrings
import pymodpkg/pyarrayobject
import pymodpkg/pyobject


proc copyAndResizeAndPrintShape*(input_arr: ptr PyArrayObject): ptr PyArrayObject
    {. exportpy .} =
  docstring"""Create some PyArrayObjects internally; return only one of them.

  This function exists to help test & debug the reference-count management.
  """
  echo("refcount(input_arr) = ", $getPyRefCnt(input_arr))
  echo("input_arr.shape = ", $input_arr.shape)
  echo("input_arr.dtype = ", $input_arr.dtype)

  result = input_arr.copy()
  echo("\nrefcount(result) = ", $getPyRefCnt(result))
  echo("result.shape = ", $result.shape)
  echo("result.dtype = ", $result.dtype)

  result.doResizeDataInplace([5, 4])
  echo("\nrefcount(result) = ", $getPyRefCnt(result))
  echo("result.shape = ", $result.shape)
  echo("result.dtype = ", $result.dtype)


proc copyAndResizeNumRowsAndPrintShape*(input_arr: ptr PyArrayObject, newNumRows: int): ptr PyArrayObject
    {. exportpy .} =
  docstring"""Create some PyArrayObjects internally; return only one of them.

  This function exists to help test & debug the reference-count management.
  """
  echo("refcount(input_arr) = ", $getPyRefCnt(input_arr))
  echo("input_arr.shape = ", $input_arr.shape)
  echo("input_arr.dtype = ", $input_arr.dtype)

  result = input_arr.copy()
  echo("\nrefcount(result) = ", $getPyRefCnt(result))
  echo("result.shape = ", $result.shape)
  echo("result.dtype = ", $result.dtype)

  result.doResizeDataInplaceNumRows(newNumRows)
  echo("\nrefcount(result) = ", $getPyRefCnt(result))
  echo("result.shape = ", $result.shape)
  echo("result.dtype = ", $result.dtype)


initPyModule("_doresize", copyAndResizeAndPrintShape, copyAndResizeNumRowsAndPrintShape)
