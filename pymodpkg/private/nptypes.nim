# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

## Definition of `NpType`, the official enum of dtypes that Pymod supports.
##
## `NpType` is the intersection of Nim types with Numpy dtypes.
##
## Documentation here:
##  http://docs.scipy.org/doc/numpy/reference/c-api.config.html
##  http://docs.scipy.org/doc/numpy/reference/c-api.dtype.html


## This is the type class of Nim types that are compatible with Numpy dtypes.
type NumpyCompatibleNimType* = bool or
    int8 or int16 or int32 or int64 or
    uint8 or uint16 or uint32 or uint64 or
    float32 or float64


# These are guaranteed to be contiguous, so NpType will be an ordinal type:
#  http://nim-lang.org/manual.html#ordinal-types
#  http://nim-lang.org/manual.html#enumeration-types
#
# NOTE:  The ordering of these enum fields doesn't match the ordering of the
# corresponding enum fields in `enum NpyTypeNums`.  There are missing fields
# in this list (for example, there is nothing corresponding to `NPY_LONGLONG`
# because we don't support the C `long long` type), and in fact I've actually
# re-ordered the elements so that all the signed types occur before all the
# unsigned types (which enables Nim subranges like `np_uint8..np_uint64`).
type NpType* = enum
  np_bool = "numpy.bool",
  np_int8 = "numpy.int8",
  np_int16 = "numpy.int16",
  np_int32 = "numpy.int32",
  np_int64 = "numpy.int64",
  np_uint8 = "numpy.uint8",
  np_uint16 = "numpy.uint16",
  np_uint32 = "numpy.uint32",
  np_uint64 = "numpy.uint64",
  np_float32 = "numpy.float32",
  np_float64 = "numpy.float64",


# I want to check the (actual, post-substitution) type of the supplied generic
# type parameter `T` at compile time, and warn (at compile time) if it's not
# a type that is valid as a Numpy dtype.
#
# We can't use a macro, because we just get the pre-substitution generic type
# parameter `T`:
#   Ident !"T"
#
# Likewise, `T.name` doesn't work -- we just get `Ident !"T"` followed by
# `Sym "name"`.
#
# Can't use `when ... elif ... elif` to specify-at-compile-time some code that
# does what we want, because when's argument can only be a constant expression,
# which a comparison involving a typedesc proc/template parameter does not
# appear to be.  (I suppose `when` statements must be evaluated BEFORE templates
# and macros.  This would make sense, because I've relied upon `when` statements
# in other places to disable my Pymod macros.)
#
# I've at least confirmed that templates can take parameters of type `typedesc`
# (in addition to `expr`, `stmt` or any normal types):
#   The parameters' types can be ordinary types or the meta types expr (stands
#   for expression), stmt (stands for statement) or typedesc (stands for type
#   description).
#    -- http://nim-lang.org/tut2.html#templates
#
# It's not encouraging that the Nim Manual describes templates as "a simple
# form of a macro", "that operates on Nim's abstract syntax trees", and later,
# as "a hygienic macro".  This suggests that templates have the exact same
# limitations as macros.
#
# Some promising-looking possibilities:
#  http://nim-lang.org/manual.html#type-classes
#  http://nim-lang.org/manual.html#user-defined-type-classes
#  http://nim-lang.org/manual.html#static-t
#  http://nim-lang.org/manual.html#typedesc
#  http://nim-lang.org/manual.html#is-operator
#  http://nim-lang.org/manual.html#converters
#  http://nim-lang.org/manual.html#convertible-relation
#
# and as a last resort, perhaps term-rewriting macros:
#  http://nim-lang.org/manual.html#term-rewriting-macros
#  http://nim-lang.org/manual.html#parameter-constraints


template toNpType*(nim_type: typedesc[bool]): NpType = np_bool
template toNpType*(nim_type: typedesc[int8]): NpType = np_int8
template toNpType*(nim_type: typedesc[int16]): NpType = np_int16
template toNpType*(nim_type: typedesc[int32]): NpType = np_int32
template toNpType*(nim_type: typedesc[int64]): NpType = np_int64
template toNpType*(nim_type: typedesc[uint8]): NpType = np_uint8
template toNpType*(nim_type: typedesc[uint16]): NpType = np_uint16
template toNpType*(nim_type: typedesc[uint32]): NpType = np_uint32
template toNpType*(nim_type: typedesc[uint64]): NpType = np_uint64
template toNpType*(nim_type: typedesc[float32]): NpType = np_float32
template toNpType*(nim_type: typedesc[float64]): NpType = np_float64

