# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

## An enumeration type indicating the element order that an array should be
## interpreted in.  When a brand new array is created, generally only NPY_CORDER
## and NPY_FORTRANORDER are used, whereas when one or more inputs are provided,
## the order can be based on them.
##  http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.NPY_ORDER
##
## NOTE:  These are alternatives.  They should not be combined.
#
# These values were hard-coded manually, corresponding with the bit-flags
# in <numpy/ndarraytypes.h>.
#
# FIXME:  We should probably determine these at compile-time, rather than
# hard-coding them, in case the bit-flag values change.
# TODO:  How can I read the #define C macro-constants into this Nim file?
type NpyOrderAlternatives* {. pure .} = enum
  anyorder     = -1,  # defined in <numpy/ndarraytypes.h>
  corder       = 0,
  fortranorder = 1,
  keeporder    = 2


## An enumeration type indicating how permissive data conversions should be.
##  http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.NPY_CASTING
##
## NOTE:  These are alternatives.  They should not be combined.
#
# These values were hard-coded manually, corresponding with the bit-flags
# in <numpy/ndarraytypes.h>.
#
# FIXME:  We should probably determine these at compile-time, rather than
# hard-coding them, in case the bit-flag values change.
# TODO:  How can I read the #define C macro-constants into this Nim file?
type NpyCastingAlternatives* {. pure .} = enum
  # Only allow identical types.
  no_casting        = 0,  # defined in <numpy/ndarraytypes.h>
  # Allow identical and casts involving byte swapping.
  equiv_casting     = 1,
  # Only allow casts which will not cause values to be rounded, truncated,
  # or otherwise changed.
  safe_casting      = 2,
  # Allow any safe casts, and casts between types of the same kind.
  # For example, float64 -> float32 is permitted with this rule.
  # (But float32 -> uint8 is NOT permitted.  - JB)
  same_kind_casting = 3,
  # Allow any cast, no matter what kind of data loss may occur.
  unsafe_casting    = 4

