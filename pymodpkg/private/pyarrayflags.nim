# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import pymodpkg/private/pyarrayobjecttype

## Array flags:
##  http://docs.scipy.org/doc/numpy/reference/c-api.array.html#array-flags
## also:
##  http://docs.scipy.org/doc/numpy/reference/c-api.array.html#flag-like-constants
## also:
##  http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_FromAny
#
# These values were hard-coded manually, corresponding with the bit-flags
# in <numpy/ndarraytypes.h>.
#
# FIXME:  We should probably determine these at compile-time, rather than
# hard-coding them, in case the bit-flag values change.
# TODO:  How can I read the #define C macro-constants into this Nim file?
type NpyArrayFlagBitValues {. pure .} = enum
  c_contiguous    = 0x0001,  # defined in <numpy/ndarraytypes.h>
  f_contiguous    = 0x0002,
  owndata         = 0x0004,
  forcecast       = 0x0010,  # "An array never has this."
  ensurecopy      = 0x0020,  # "An array never has this."
  ensurearray     = 0x0040,  # "An array never has this."
  elementstrides  = 0x0080,  # "An array never has this."
  aligned         = 0x0100,
  notswapped      = 0x0200,
  writeable       = 0x0400,
  updateifcopy    = 0x1000


## http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.NPY_ARRAY_BEHAVED
## = NPY_ARRAY_ALIGNED | NPY_ARRAY_WRITEABLE
const NPY_ARRAY_BEHAVED*: cint =
    ord(NpyArrayFlagBitValues.aligned) and ord(NpyArrayFlagBitValues.writeable)

## http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.NPY_ARRAY_CARRAY
## = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_BEHAVED
const NPY_ARRAY_CARRAY*: cint =
    ord(NpyArrayFlagBitValues.c_contiguous) and NPY_ARRAY_BEHAVED

## http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.NPY_ARRAY_CARRAY_RO
## = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED
const NPY_ARRAY_CARRAY_RO*: cint =
    ord(NpyArrayFlagBitValues.c_contiguous) and ord(NpyArrayFlagBitValues.aligned)


type PyArrayFlags* {. pure .} = enum
  c_contiguous,
  f_contiguous,
  owndata,
  forcecast,
  ensurecopy,
  ensurearray,
  elementstrides,
  aligned,
  notswapped,
  writeable,
  updateifcopy


template flagBitIsOn*(xflagval: cint, xflagname: expr): expr {. immediate .} =
  ((xflagval and ord(NpyArrayFlagBitValues.xflagname)) != 0)

template setFlag(s: var set[PyArrayFlags], flagval: cint, flagname: expr): stmt
    {. immediate .} =
  if flagBitIsOn(flagval, flagname): incl(s, PyArrayFlags.flagname)

proc flags*(arr: ptr PyArrayObject): set[PyArrayFlags] =
  ## NOTE:  The return value from this proc is a newly-created set instance.
  ## It is NOT synchronised with the value of the `flags` attribute in the
  ## PyArrayObject.
  ##
  ## If you modify the members of the set, the value in the `flags` attribute
  ## will NOT be updated automatically; likewise, if you modify the value of
  ## the `flags` attribute, the members of the set will NOT be updated.
  let flagval: cint = getFLAGS(arr)
  result = {}
  setFlag(result, flagval, c_contiguous)
  setFlag(result, flagval, f_contiguous)
  setFlag(result, flagval, owndata)
  setFlag(result, flagval, forcecast)
  setFlag(result, flagval, ensurecopy)
  setFlag(result, flagval, ensurearray)
  setFlag(result, flagval, elementstrides)
  setFlag(result, flagval, aligned)
  setFlag(result, flagval, notswapped)
  setFlag(result, flagval, writeable)
  setFlag(result, flagval, updateifcopy)

