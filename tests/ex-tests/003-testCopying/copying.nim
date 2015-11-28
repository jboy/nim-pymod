# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import strutils

import pymod
import pymodpkg/docstrings
import pymodpkg/miscutils
import pymodpkg/pyarrayobject


proc copyArray*(in_arr: ptr PyArrayObject): ptr PyArrayObject {. exportpy .} =
  docstring"""Copy `in_arr` and verify that the copy has a new data array."""
  let p1: pointer = in_arr.data
  let out_arr: ptr PyArrayObject = in_arr.copy()
  let p2: pointer = in_arr.data
  assert(p1 == p2)
  let p3: pointer = out_arr.data
  assert(p1 != p3)
  assert(in_arr != out_arr)
  return out_arr


proc asTypeArray*(in_arr: ptr PyArrayObject, newtype: cint): ptr PyArrayObject
    {. exportpy .} =
  docstring"""Use `.asType` to change the dtype of `in_arr` to `newtype`."""
  let p1: pointer = in_arr.data
  let t1: cint = in_arr.getTYPE()
  let tn1: cint = in_arr.getDESCR().type_num
  assert(t1 == tn1)

  let nptype_newtype: NpType = newtype.toNpType()
  let out_arr: ptr PyArrayObject = in_arr.createAsTypeNewData(nptype_newtype)

  let p2: pointer = in_arr.data
  assert(p1 == p2)
  let t2: cint = in_arr.getTYPE()
  let tn2: cint = in_arr.getDESCR().type_num
  assert(t2 == tn2)
  assert(t1 == t2)

  let p3: pointer = out_arr.data
  let t3: cint = out_arr.getTYPE()
  let tn3: cint = out_arr.getDESCR().type_num
  assert(t3 == tn3)
  assert(t3 == newtype)

  if newtype == t1:
    assert(t1 == t3)
  else:
    assert(t1 != t3)

  assert(p1 != p3)
  assert(in_arr != out_arr)

  return out_arr


initPyModule("_copying", copyArray, asTypeArray)
