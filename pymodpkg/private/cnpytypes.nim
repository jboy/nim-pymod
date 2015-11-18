# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

## Definition of `NpyTypeNums`, a Nim wrapper enum for the C enum `NPY_TYPES`
## in the Numpy C-API Data Type API.
##
## We only wrap the `NPY_TYPES` enum fields that Pymod supports.
##
## Documentation here:
##  http://docs.scipy.org/doc/numpy/reference/c-api.config.html
##  http://docs.scipy.org/doc/numpy/reference/c-api.dtype.html


import strutils  # `%`
import pymodpkg/private/nptypes


# These fields were extracted from `enum NPY_TYPES` in <numpy/ndarraytypes.h>.
# The Nim enum fields are ordered to match the enum fields in `enum NPY_TYPES`
# (so don't add or delete any of these enum constants -- not even the ones we
# don't support).
#
# (I wanted to `importc` the fields directly from C as either some const values
# or an enum, but I couldn't work out how to do that -- neither directly from
# the Numpy header, nor from my own C header/source file.)
#
# FIXME:  Work out how to load/mirror this enum directly from the C header.
#
# Also potentially of interest: enum NPY_TYPECHAR in the same header.
type NpyTypeNums* {. pure .} = enum
  NPY_BOOL = 0,
  NPY_BYTE,
  NPY_UBYTE,
  NPY_SHORT,
  NPY_USHORT,
  NPY_INT,
  NPY_UINT,
  NPY_LONG,
  NPY_ULONG,
  NPY_LONGLONG,  # we don't support this
  NPY_ULONGLONG,  # we don't support this
  NPY_FLOAT,
  NPY_DOUBLE,
  # There are more, but we don't support them.


# FIXME:  Hard-coding this C-type-to-sized-type mapping is NOT the right way
# to do this.
#
# Instead, we should work it out at compile-time, like in <numpy/npy_common.h>,
# perhaps using the #defined constants (like `NPY_INT32`) in that header.
# (But again, I can't work out how to `importc` enum/const/#defined constants.)
#
# See also:
#  http://docs.scipy.org/doc/numpy/reference/c-api.config.html
#  http://docs.scipy.org/doc/numpy/reference/c-api.dtype.html
#
# Perhaps we should use a `when` statement:
#  http://nim-lang.org/manual.html#when-statement
# and/or a template:
#  http://nim-lang.org/manual.html#templates
# with the sizeof proc:
#  http://nim-lang.org/manual.html#reference-and-pointer-types
proc toNpType*(type_num: cint): NpType =
  case type_num
  of ord(NpyTypeNums.NPY_BOOL): result = np_bool
  of ord(NpyTypeNums.NPY_BYTE): result = np_int8
  of ord(NpyTypeNums.NPY_UBYTE): result = np_uint8
  of ord(NpyTypeNums.NPY_SHORT): result = np_int16
  of ord(NpyTypeNums.NPY_USHORT): result = np_uint16
  of ord(NpyTypeNums.NPY_INT): result = np_int32
  of ord(NpyTypeNums.NPY_UINT): result = np_uint32
  of ord(NpyTypeNums.NPY_LONG): result = np_int64
  of ord(NpyTypeNums.NPY_ULONG): result = np_uint64
  of ord(NpyTypeNums.NPY_FLOAT): result = np_float32
  of ord(NpyTypeNums.NPY_DOUBLE): result = np_float64
  else:
    let msg = "Numpy type num $1 is not supported" % $type_num
    raise newException(ValueError, msg)


proc toNpyTypeNums*(nptype: NpType): NpyTypeNums {. inline, nosideEffect .} =
  case nptype
  of np_bool: result = NpyTypeNums.NPY_BOOL
  of np_int8: result = NpyTypeNums.NPY_BYTE
  of np_int16: result = NpyTypeNums.NPY_SHORT
  of np_int32: result = NpyTypeNums.NPY_INT
  of np_int64: result = NpyTypeNums.NPY_LONG
  of np_uint8: result = NpyTypeNums.NPY_UBYTE
  of np_uint16: result = NpyTypeNums.NPY_USHORT
  of np_uint32: result = NpyTypeNums.NPY_UINT
  of np_uint64: result = NpyTypeNums.NPY_ULONG
  of np_float32: result = NpyTypeNums.NPY_FLOAT
  of np_float64: result = NpyTypeNums.NPY_DOUBLE

