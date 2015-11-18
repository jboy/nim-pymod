# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

## These are C++-style iterators to iterate over instances of PyArrayObject.
## They don't correspond to any types or functions in the Numpy C-API.

const doIterRangeChecks: bool = not defined(release)

import strutils
import typetraits  # name(t: typedesc)

import pymodpkg/miscutils
import pymodpkg/ptrutils

import pymodpkg/private/nptypes
import pymodpkg/private/pyarrayobjecttype


proc getLowHighBounds[NT](a: ptr PyArrayObject):
    tuple[low: ptr NT, high: ptr NT] {. inline .} =
  let low: ptr NT = a.data(NT)
  let num_elems = a.elcount

  # Subtract 1 to get the Nim-idiom `high` position (the highest valid data position),
  # rather than the usual C-idiom of "1 beyond the highest valid data position".
  let high = offset_ptr(low, num_elems - 1)
  result = (low, high)


type PyArrayForwardIterator*[T] = object
  ## An iterator that can only move forward incrementally.
  ##
  ## Inspired by ye olde C++ STL ForwardIterator type:
  ##  https://stdcxx.apache.org/doc/stdlibug/2-2.html
  ##  http://www.cplusplus.com/reference/iterator/ForwardIterator/
  ##
  ## We use a double range-checking system:  The iterator will check its own
  ## bounds whenever an attempt is made to dereference the iterator;  if the
  ## iterator is outside of its own bounds, a RangeError exception will be
  ## raised (which will become an IndexError in Python, natch).
  ##
  ## However, these internal range checks will be disabled in release builds.
  ##
  ## Hence, if necessary, the user can/should perform iterator range-checking
  ## using the supplied PyArrayIteratorBounds:
  ##
  ##   let bounds = arr.getBounds(int32)
  ##   var iter = arr.iterateForward(int32)
  ##   while iter in bounds:
  ##     doSomethingWith(iter[])
  ##     inc(iter)
  ##
  pos: ptr T
  when doIterRangeChecks:
    # Check ranges.  Catch mistakes.
    low: ptr T
    high: ptr T


proc initPyArrayForwardIterator*[T](arr: ptr PyArrayObject):
    PyArrayForwardIterator[T] {. inline .} =
  let (low, high) = getLowHighBounds[T](arr)
  when doIterRangeChecks:
    # Check ranges.  Catch mistakes.
    result = PyArrayForwardIterator[T](pos: low, low: low, high: high)
  else:
    result = PyArrayForwardIterator[T](pos: low)


when doIterRangeChecks:
  # Check ranges.  Catch mistakes.

  template isNotValid[T](fi: PyArrayForwardIterator[T]): bool =
    ## Test whether the PyArrayForwardIterator is outside of its valid bounds.
    ## This range-checking will be disabled in release builds.
    (fi.pos < fi.low or fi.pos > fi.high)


  proc assertValid[T](fi: PyArrayForwardIterator[T]) =
    # Note: Use a proc rather than a template, to get a fuller stack trace.
    if fi.isNotValid:
      let msg = "PyArrayForwardIterator[$1] dereferenced at pos $2, out of bounds [$3, $4], with sizeof($1) == $5" %
          [getCompileTimeType(T), fi.pos.toHex, fi.low.toHex, fi.high.toHex, $sizeof(T)]
      raise newException(RangeError, msg)

else:
  # Don't check ranges.  Fail fast, fail forward.
  template assertValid[T](fi: PyArrayForwardIterator[T]) =
    # Note: Use a template rather than a proc, to ensure it will disappear.
    discard


when doIterRangeChecks:
  # Check ranges.  Catch mistakes.

  proc `[]`*[T](fi: PyArrayForwardIterator[T]): var T =
    assertValid(fi)
    return fi.pos[]

  proc `[]=`*[T](fi: PyArrayForwardIterator[T], val: T) =
    assertValid(fi)
    fi.pos[] = val

  proc inc*[T](fi: var PyArrayForwardIterator[T]) {. inline .} =
    fi.pos = offset_ptr(fi.pos)

else:
  template `[]`*[T](fi: PyArrayForwardIterator[T]): var T =
    (fi.pos[])

  template `[]=`*[T](fi: PyArrayForwardIterator[T], val: T): stmt =
    (fi.pos[] = val)

  proc inc*[T](fi: var PyArrayForwardIterator[T]) {. inline .} =
    fi.pos = cast[ptr T](offset_void_ptr_in_bytes(fi.pos, sizeof(T)))


template `==`*[T](lhs, rhs: PyArrayForwardIterator[T]): bool =
  ## Note:  If possible, use the (iter in bounds) idiom instead.
  (lhs.pos == rhs.pos)


template `!=`*[T](lhs, rhs: PyArrayForwardIterator[T]): bool =
  ## Note:  If possible, use the (iter in bounds) idiom instead.
  (lhs.pos != rhs.pos)


template `<=`*[T](lhs, rhs: PyArrayForwardIterator[T]): bool =
  ## Note:  If possible, use the (iter in bounds) idiom instead.
  (cast[int](lhs.pos) <= cast[int](rhs.pos))


template `<`*[T](lhs, rhs: PyArrayForwardIterator[T]): bool =
  ## Note:  If possible, use the (iter in bounds) idiom instead.
  (cast[int](lhs.pos) < cast[int](rhs.pos))


type PyArrayRandomAccessIterator*[T] = object
  ## An iterator that can jump forward or backward by any arbitrary amount;
  ## effectively a pointer with range checking.
  ##
  ## Inspired by ye olde C++ STL RandomAccessIterator type:
  ##  https://stdcxx.apache.org/doc/stdlibug/2-2.html
  ##  http://www.cplusplus.com/reference/iterator/RandomAccessIterator/
  ##
  ## We use a double range-checking system:  The iterator will check its own
  ## bounds whenever an attempt is made to dereference the iterator;  if the
  ## iterator is outside of its own bounds, a RangeError exception will be
  ## raised (which will become an IndexError in Python, natch).
  ##
  ## However, these internal range checks will be disabled in release builds.
  ##
  ## Hence, if necessary, the user can/should perform iterator range-checking
  ## using the supplied PyArrayIteratorBounds:
  ##
  ##   let bounds = arr.getBounds(int32)
  ##   var iter = arr.accessFlat(int32)
  ##   while iter in bounds:
  ##     doSomethingWith(iter[])
  ##     inc(iter)
  ##
  pos: ptr T
  when doIterRangeChecks:
    # Check ranges.  Catch mistakes.
    low: ptr T
    high: ptr T


proc initPyArrayRandomAccessIterator*[T](arr: ptr PyArrayObject):
    PyArrayRandomAccessIterator[T] {. inline .} =
  let (low, high) = getLowHighBounds[T](arr)
  when doIterRangeChecks:
    # Check ranges.  Catch mistakes.
    result = PyArrayRandomAccessIterator[T](pos: low, low: low, high: high)
  else:
    result = PyArrayRandomAccessIterator[T](pos: low)


when doIterRangeChecks:
  # Check ranges.  Catch mistakes.

  template isNotValid[T](
      rai: PyArrayRandomAccessIterator[T], pos: ptr T): bool =
    ## Test whether the PyArrayRandomAccessIterator is outside of its valid bounds.
    ## This range-checking will be disabled in release builds.
    (pos < rai.low or pos > rai.high)


  proc assertValid[T](rai: PyArrayRandomAccessIterator[T]) =
    # Note: Use a proc rather than a template, to get a fuller stack trace.
    if rai.isNotValid(rai.pos):
      let itertype = rai.getGenericTypeName
      let iterdescr = "$1[$2](sizeof($1)=$3" %
          [itertype, getCompileTimeType(T), $sizeof(T)]
      let msg = "$1 dereferenced at pos $2, out of bounds [$3, $4]" %
          [iterdescr, rai.pos.toHex, rai.low.toHex, rai.high.toHex]
      raise newException(RangeError, msg)


  proc assertValid[T](
      rai: PyArrayRandomAccessIterator[T], offset_pos: ptr T) =
    # Note: Use a proc rather than a template, to get a fuller stack trace.
    if rai.isNotValid(offset_pos):
      let itertype = rai.getGenericTypeName
      let iterdescr = "$1[$2](sizeof($1)=$3" %
          [itertype, getCompileTimeType(T), $sizeof(T)]
      let msg = "$1 dereferenced at pos $2, out of bounds [$3, $4]" %
          [iterdescr, offset_pos.toHex, rai.low.toHex, rai.high.toHex]
      raise newException(RangeError, msg)

else:
  # Don't check ranges.  Random access w/o range checks: What could go wrong?
  template assertValid[T](rai: PyArrayRandomAccessIterator[T]) =
    # Note: Use a template rather than a proc, to ensure it will disappear.
    discard

  template assertValid[T](
      rai: PyArrayRandomAccessIterator[T], offset_pos: ptr T) =
    # Note: Use a template rather than a proc, to ensure it will disappear.
    discard


when doIterRangeChecks:
  # Check ranges.  Catch mistakes.

  proc `[]`*[T](rai: PyArrayRandomAccessIterator[T]): var T =
    assertValid(rai)
    return rai.pos[]


  proc `[]=`*[T](rai: PyArrayRandomAccessIterator[T], val: T) =
    assertValid(rai)
    rai.pos[] = val

else:
  template `[]`*[T](rai: PyArrayRandomAccessIterator[T]): var T =
    (rai.pos[])

  template `[]=`*[T](rai: PyArrayRandomAccessIterator[T], val: T): stmt =
    (rai.pos[] = val)


proc `[]`*[T](rai: PyArrayRandomAccessIterator[T], idx: int): var T =
  let offset_pos = offset_ptr(rai.pos, idx)
  assertValid(rai, offset_pos)
  return offset_pos[]


proc `[]=`*[T](rai: PyArrayRandomAccessIterator[T], idx: int, val: T) =
  let offset_pos = offset_ptr(rai.pos, idx)
  assertValid(rai, offset_pos)
  offset_pos[] = val


proc inc*[T](rai: var PyArrayRandomAccessIterator[T], delta: int) {. inline .} =
  rai.pos = offset_ptr(rai.pos, delta)


when doIterRangeChecks:
  proc inc*[T](rai: var PyArrayRandomAccessIterator[T]) {. inline .} =
    rai.pos = offset_ptr(rai.pos)
else:
  proc inc*[T](rai: var PyArrayRandomAccessIterator[T]) {. inline .} =
    rai.pos = cast[ptr T](offset_void_ptr_in_bytes(rai.pos, sizeof(T)))


proc dec*[T](rai: var PyArrayRandomAccessIterator[T], delta: int) {. inline .} =
  rai.pos = offset_ptr(rai.pos, -delta)


when doIterRangeChecks:
  proc dec*[T](rai: var PyArrayRandomAccessIterator[T]) {. inline .} =
    rai.pos = offset_ptr(rai.pos, -1)
else:
  proc dec*[T](rai: var PyArrayRandomAccessIterator[T]) {. inline .} =
    rai.pos = cast[ptr T](offset_void_ptr_in_bytes(rai.pos, -sizeof(T)))


proc `+`*[T](rai: PyArrayRandomAccessIterator[T], delta: int):
    PyArrayRandomAccessIterator[T] {. inline .} =
  result.pos = offset_ptr(rai.pos, delta)
  when doIterRangeChecks:
    # Check ranges.  Catch mistakes.
    result.low = rai.low
    result.high = rai.high


proc `-`*[T](rai: PyArrayRandomAccessIterator[T], delta: int):
    PyArrayRandomAccessIterator[T] {. inline .} =
  result.pos = offset_ptr(rai.pos, -delta)
  when doIterRangeChecks:
    # Check ranges.  Catch mistakes.
    result.low = rai.low
    result.high = rai.high


template `==`*[T](lhs, rhs: PyArrayRandomAccessIterator[T]): bool =
  ## Note:  If possible, use the (iter in bounds) idiom instead.
  (lhs.pos == rhs.pos)


template `!=`*[T](lhs, rhs: PyArrayRandomAccessIterator[T]): bool =
  ## Note:  If possible, use the (iter in bounds) idiom instead.
  (lhs.pos != rhs.pos)


template `<=`*[T](lhs, rhs: PyArrayRandomAccessIterator[T]): bool =
  ## Note:  If possible, use the (iter in bounds) idiom instead.
  (cast[int](lhs.pos) <= cast[int](rhs.pos))


template `<`*[T](lhs, rhs: PyArrayRandomAccessIterator[T]): bool =
  ## Note:  If possible, use the (iter in bounds) idiom instead.
  (cast[int](lhs.pos) < cast[int](rhs.pos))


type PyArrayIteratorBounds*[T] = object
  low: ptr T
  high: ptr T


proc initPyArrayIteratorBounds*[T](arr: ptr PyArrayObject):
    PyArrayIteratorBounds[T] {. inline .} =
  let (low, high) = getLowHighBounds[T](arr)
  result = PyArrayIteratorBounds[T](low: low, high: high)


template contains*[T](
    bounds: PyArrayIteratorBounds[T],
    fi: PyArrayForwardIterator[T]): bool =
  ## Test whether the PyArrayForwardIterator is within its bounds.
  ##
  ## This is intended to be used by user code (in contrast to `isNotValid`,
  ## which is not intended to be used by user code; it is for range-checking
  ## that will be disabled in release builds).
  (fi.pos <= bounds.high)


template contains*[T](
    bounds: PyArrayIteratorBounds[T],
    rai: PyArrayRandomAccessIterator[T]): bool =
  ## Test whether the PyArrayRandomAccessIterator is within its bounds.
  ##
  ## This is intended to be used by user code (in contrast to `isNotValid`,
  ## which is not intended to be used by user code; it is for range-checking
  ## that will be disabled in release builds).
  (bounds.low <= rai.pos and rai.pos <= bounds.high)

