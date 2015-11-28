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


proc createsome*(input_arr: ptr PyArrayObject): ptr PyArrayObject
    {. exportpy .} =
  docstring"""Create some PyArrayObjects internally; return only one of them.

  This function exists to help test & debug the reference-count management.
  """
  echo("refcount(input_arr) = ", $getPyRefCnt(input_arr))

  let tmp1_arr: ptr PyArrayObject = createSimpleNew(input_arr.dimensions, np_int32)
  echo("refcount(tmp1_arr) = ", $getPyRefCnt(tmp1_arr))

  let out_arr: ptr PyArrayObject = createSimpleNew(input_arr.dimensions, np_int32)
  doFILLWBYTE(out_arr, 0)
  echo("refcount(out_arr) = ", $getPyRefCnt(out_arr))

  let tmp2_arr: ptr PyArrayObject = createSimpleNew(input_arr.dimensions, np_int32)
  echo("refcount(tmp2_arr) = ", $getPyRefCnt(tmp2_arr))

  return out_arr


proc identity*(arr: ptr PyArrayObject): ptr PyArrayObject {. exportpy .} =
  docstring"""Return the function argument immediately, unchanged.

  This function exists to help test & debug the reference-count management.
  """
  echo("refcount(arr) = ", $getPyRefCnt(arr))
  return arr


proc twogoinonecomesout*(arr1, arr2: ptr PyArrayObject): ptr PyArrayObject {. exportpy .} =
  docstring"""Two PyArrayObjects go in, and only one comes out.

  This function exists to help test & debug the reference-count management.
  """
  echo("refcount(arr1) = ", $getPyRefCnt(arr1))
  echo("refcount(arr2) = ", $getPyRefCnt(arr2))
  return arr2


initPyModule("_refcount", createsome, identity, twogoinonecomesout)
