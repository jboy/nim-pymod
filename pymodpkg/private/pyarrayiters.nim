# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

## These are C++-style iterators to iterate over instances of PyArrayObject.
## They don't correspond to any types or functions in the Numpy C-API.

const doWithinRangeChecks: bool = not defined(release)
const doSamePyArrayChecks: bool = not defined(release)

import strutils
import typetraits  # name(t: typedesc)

import pymodpkg/miscutils
import pymodpkg/ptrutils

import pymodpkg/private/nptypes
import pymodpkg/private/pyarrayobjecttype


proc getLowHighBounds[NT](a: ptr PyArrayObject):
    tuple[low: ptr NT, high: ptr NT, numElems: int] {. inline .} =
  let low: ptr NT = a.data(NT)
  let numElems: int = int(a.elcount)
  # We subtract 1 from `num_elems` to get the Nim-idiom `high` position
  # (the highest valid data position), rather than the usual C-idiom of
  # "1 beyond the highest valid data position".
  let high = offset_ptr(low, numElems - 1)
  result = (low, high, numElems)


type PyArrayIterBounds*[T] = object
  arr: ptr PyArrayObject
  low: ptr T
  high: ptr T
  numElems: int


proc initPyArrayIterBounds*[T](arr: ptr PyArrayObject):
    PyArrayIterBounds[T] {. inline .} =
  let (low, high, numElems) = getLowHighBounds[T](arr)
  result = PyArrayIterBounds[T](arr: arr, low: low, high: high, numElems: numElems)

proc numElems*[T](bounds: PyArrayIterBounds[T]): int {. inline .} = bounds.numElems


type PyArrayForwardIter*[T] = object
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
  ## using the supplied PyArrayIterBounds:
  ##
  ##   let bounds = arr.getBounds(int32)
  ##   var iter = arr.iterateFlat(int32)
  ##   while iter in bounds:
  ##     doSomethingWith(iter[])
  ##     inc(iter)
  ##
  pos: ptr T
  arr: ptr PyArrayObject
  when doWithinRangeChecks:
    # Check ranges.  Catch mistakes.
    low: ptr T
    high: ptr T


proc initPyArrayForwardIter*[T](arr: ptr PyArrayObject):
    PyArrayForwardIter[T] {. inline .} =
  let (low, high, _) = getLowHighBounds[T](arr)
  when doWithinRangeChecks:
    # Check ranges.  Catch mistakes.
    result = PyArrayForwardIter[T](pos: low, arr: arr, low: low, high: high)
  else:
    discard high
    result = PyArrayForwardIter[T](pos: low, arr: arr)


when doWithinRangeChecks:
  # Check ranges.  Catch mistakes.

  template isNotWithinRange[T](fi: PyArrayForwardIter[T]): bool =
    ## Test whether the PyArrayForwardIter is outside of its valid bounds.
    (fi.pos < fi.low or fi.pos > fi.high)

  proc assertWithinRange[T](fi: PyArrayForwardIter[T]) =
    ## Assert that the PyArrayForwardIter is within its valid bounds when
    ## dereferenced.  This bounds-checking will be disabled in release builds.
    # Note: Use a proc rather than a template, to get a fuller stack trace.
    if isNotWithinRange(fi):
      let itertype = fi.getGenericTypeName
      let msg = "$1[$2] dereferenced at pos $3, out of bounds [$4, $5], with sizeof($2) == $6" %
          [itertype, getCompileTimeType(T),
              fi.pos.toHex, fi.low.toHex, fi.high.toHex, $sizeof(T)]
      raise newException(RangeError, msg)

  proc `[]`*[T](fi: PyArrayForwardIter[T]): var T =
    assertWithinRange(fi)
    return fi.pos[]

  proc `[]=`*[T](fi: PyArrayForwardIter[T], val: T) =
    assertWithinRange(fi)
    fi.pos[] = val


  template isTooFarBeyondRange[T](fi: PyArrayForwardIter[T]): bool =
    ## Test whether the PyArrayForwardIter is more than an allowable tolerance
    ## (a step of 1x sizeof(T)) outside of its valid bounds.
    (fi.pos < fi.low or fi.pos > offset_ptr(fi.high))

  proc assertWithinToleranceOfRange[T](fi: PyArrayForwardIter[T]) =
    ## Assert that the PyArrayForwardIter is within an allowable tolerance
    ## (a step of 1x sizeof(T)) of its valid bounds when incremented.  This
    ## bounds-checking will be disabled in release builds.
    # Note: Use a proc rather than a template, to get a fuller stack trace.
    if isTooFarBeyondRange(fi):
      let itertype = fi.getGenericTypeName
      let msg = "$1[$2] has been incremented to pos $3, out of bounds [$4, $5], with sizeof($2) == $6" %
          [itertype, getCompileTimeType(T),
              fi.pos.toHex, fi.low.toHex, fi.high.toHex, $sizeof(T)]
      raise newException(RangeError, msg)


  proc inc*[T](fi: var PyArrayForwardIter[T]) {. inline .} =
    #assertWithinToleranceOfRange(fi)
    fi.pos = offset_ptr(fi.pos)

  proc derefInc*[T](fi: var PyArrayForwardIter[T]): var T {. inline .} =
    assertWithinRange(fi)
    # If it's "within bounds", then it is guaranteed to be "within tolerance of bounds".
    # So, no need to assertWithinToleranceOfRange(fi).
    let prev: ptr T = fi.pos
    fi.pos = offset_ptr(prev)
    result = prev[]

  proc incFast*[T](fi: var PyArrayForwardIter[T]; positiveDelta: Positive) {. inline .} =
    ## "Fast forward"
    assertWithinToleranceOfRange(fi)
    fi.pos = offset_ptr(fi.pos, positiveDelta)

else:
  template `[]`*[T](fi: PyArrayForwardIter[T]): var T =
    (fi.pos[])

  proc `[]=`*[T](fi: PyArrayForwardIter[T], val: T) {. inline .} =
    fi.pos[] = val

  proc inc*[T](fi: var PyArrayForwardIter[T]) {. inline .} =
    fi.pos = cast[ptr T](offset_void_ptr_in_bytes(fi.pos, sizeof(T)))

  proc derefInc*[T](fi: var PyArrayForwardIter[T]): var T {. inline .} =
    let prev: ptr T = fi.pos
    fi.pos = offset_ptr(prev)
    result = prev[]

  proc incFast*[T](fi: var PyArrayForwardIter[T]; positiveDelta: Positive) {. inline .} =
    ## "Fast forward"
    fi.pos = offset_ptr(fi.pos, positiveDelta)


when doSamePyArrayChecks:
  # Check that our iterators are pointing at the same array.

  template isNotSamePyArray[T](bounds: PyArrayIterBounds[T];
      fi: PyArrayForwardIter[T]): bool =
    (bounds.arr != fi.arr)

  proc assertSamePyArray[T](bounds: PyArrayIterBounds[T];
      fi: PyArrayForwardIter[T]) =
    # Note: Use a proc rather than a template, to get a fuller stack trace.
    if isNotSamePyArray(bounds, fi):
      let msg = "A PyArrayForwardIter[$1] was compared to a PyArrayIterBounds[$1], but they point to different PyArrayObjects" %
          getCompileTimeType(T)
      raise newException(ValueError, msg)

  proc contains*[T](bounds: PyArrayIterBounds[T],
      fi: PyArrayForwardIter[T]): bool {.inline.} =
    ## Test whether the PyArrayForwardIter is within its bounds.
    ##
    ## This is intended to be used by user code (in contrast to `isNotWithinRange`,
    ## which is not intended to be used by user code; it is for range-checking
    ## that will be disabled in release builds).
    assertSamePyArray(bounds, fi)
    (fi.pos <= bounds.high)

  template isNotSamePyArray[T](lhs, rhs: PyArrayForwardIter[T]): bool =
    (lhs.arr != rhs.arr)

  proc assertSamePyArray[T](lhs, rhs: PyArrayForwardIter[T]) =
    # Note: Use a proc rather than a template, to get a fuller stack trace.
    if isNotSamePyArray(lhs, rhs):
      let msg = "Two PyArrayForwardIter[$1] were compared, but they point to different PyArrayObjects" %
          getCompileTimeType(T)
      raise newException(ValueError, msg)

  proc `==`*[T](lhs, rhs: PyArrayForwardIter[T]): bool {.inline.} =
    ## Note:  If possible, use the (iter in bounds) idiom instead.
    assertSamePyArray(lhs, rhs)
    (lhs.pos == rhs.pos)

  proc `!=`*[T](lhs, rhs: PyArrayForwardIter[T]): bool {.inline.} =
    ## Note:  If possible, use the (iter in bounds) idiom instead.
    assertSamePyArray(lhs, rhs)
    (lhs.pos != rhs.pos)

  proc `<=`*[T](lhs, rhs: PyArrayForwardIter[T]): bool {.inline.} =
    ## Note:  If possible, use the (iter in bounds) idiom instead.
    assertSamePyArray(lhs, rhs)
    (cast[int](lhs.pos) <= cast[int](rhs.pos))

  proc `<`*[T](lhs, rhs: PyArrayForwardIter[T]): bool {.inline.} =
    ## Note:  If possible, use the (iter in bounds) idiom instead.
    assertSamePyArray(lhs, rhs)
    (cast[int](lhs.pos) < cast[int](rhs.pos))

else:

  template contains*[T](bounds: PyArrayIterBounds[T],
      fi: PyArrayForwardIter[T]): bool =
    ## Test whether the PyArrayForwardIter is within its bounds.
    ##
    ## This is intended to be used by user code (in contrast to `isNotWithinRange`,
    ## which is not intended to be used by user code; it is for range-checking
    ## that will be disabled in release builds).
    (fi.pos <= bounds.high)

  template `==`*[T](lhs, rhs: PyArrayForwardIter[T]): bool =
    ## Note:  If possible, use the (iter in bounds) idiom instead.
    (lhs.pos == rhs.pos)

  template `!=`*[T](lhs, rhs: PyArrayForwardIter[T]): bool =
    ## Note:  If possible, use the (iter in bounds) idiom instead.
    (lhs.pos != rhs.pos)

  template `<=`*[T](lhs, rhs: PyArrayForwardIter[T]): bool =
    ## Note:  If possible, use the (iter in bounds) idiom instead.
    (cast[int](lhs.pos) <= cast[int](rhs.pos))

  template `<`*[T](lhs, rhs: PyArrayForwardIter[T]): bool =
    ## Note:  If possible, use the (iter in bounds) idiom instead.
    (cast[int](lhs.pos) < cast[int](rhs.pos))


type PyArrayRandAccIter*[T] = object
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
  ## using the supplied PyArrayIterBounds:
  ##
  ##   let bounds = arr.getBounds(int32)
  ##   var iter = arr.accessFlat(int32)
  ##   while iter in bounds:
  ##     doSomethingWith(iter[])
  ##     inc(iter)
  ##
  pos: ptr T
  arr: ptr PyArrayObject
  flatstride: int  # in bytes, as strides usually are.
  when doWithinRangeChecks:
    # Check ranges.  Catch mistakes.
    low: ptr T
    high: ptr T


proc initPyArrayRandAccIter*[T](arr: ptr PyArrayObject; initOffset, incDelta: int):
    PyArrayRandAccIter[T] {. inline .} =
  let (low, high, _) = getLowHighBounds[T](arr)
  let initPos = offset_ptr(low, initOffset)
  let flatstride = incDelta * sizeof(T)
  when doWithinRangeChecks:
    # Check ranges.  Catch mistakes.
    result = PyArrayRandAccIter[T](pos: initPos, arr: arr, flatstride: flatstride,
        low: low, high: high)
  else:
    discard high
    result = PyArrayRandAccIter[T](pos: initPos, arr: arr, flatstride: flatstride)


proc elemStep*[T](rai: PyArrayRandAccIter[T]): int {. inline .} =
  ## Return the number of elements stepped per increment.
  (rai.flatstride div sizeof(T))


when doWithinRangeChecks:
  # Check ranges.  Catch mistakes.

  template isNotWithinRange[T](
      rai: PyArrayRandAccIter[T], pos: ptr T): bool =
    ## Test whether the PyArrayRandAccIter is outside of its valid bounds.
    (pos < rai.low or pos > rai.high)

  proc assertWithinRange[T](rai: PyArrayRandAccIter[T]) =
    ## Assert that the PyArrayRandAccIter is within its valid bounds when
    ## dereferenced.  This bounds-checking will be disabled in release builds.
    # Note: Use a proc rather than a template, to get a fuller stack trace.
    if isNotWithinRange(rai, rai.pos):
      let itertype = rai.getGenericTypeName
      let iterdescr = "$1[$2](sizeof($1)=$3; flatstride=$4)" %
          [itertype, getCompileTimeType(T), $sizeof(T), $rai.flatstride]
      let msg = "$1 dereferenced at pos $2, out of bounds [$3, $4]" %
          [iterdescr, rai.pos.toHex, rai.low.toHex, rai.high.toHex]
      raise newException(RangeError, msg)

  proc `[]`*[T](rai: PyArrayRandAccIter[T]): var T =
    assertWithinRange(rai)
    return rai.pos[]

  proc `[]=`*[T](rai: PyArrayRandAccIter[T], val: T) =
    assertWithinRange(rai)
    rai.pos[] = val


  proc assertWithinRange[T](
      rai: PyArrayRandAccIter[T], offset_pos: ptr T) =
    ## Assert that the PyArrayRandAccIter is within its valid bounds when
    ## dereferenced.  This bounds-checking will be disabled in release builds.
    # Note: Use a proc rather than a template, to get a fuller stack trace.
    if isNotWithinRange(rai, offset_pos):
      let itertype = rai.getGenericTypeName
      let iterdescr = "$1[$2](sizeof($1)=$3; flatstride=$4)" %
          [itertype, getCompileTimeType(T), $sizeof(T), $rai.flatstride]
      let msg = "$1 dereferenced at pos $2, out of bounds [$3, $4]" %
          [iterdescr, offset_pos.toHex, rai.low.toHex, rai.high.toHex]
      raise newException(RangeError, msg)

  proc `[]`*[T](rai: PyArrayRandAccIter[T], idx: int): var T =
    let offset_pos = offset_ptr_in_bytes(rai.pos, idx * rai.flatstride)
    assertWithinRange(rai, offset_pos)
    return offset_pos[]

  proc `[]=`*[T](rai: PyArrayRandAccIter[T], idx: int, val: T) =
    let offset_pos = offset_ptr_in_bytes(rai.pos, idx * rai.flatstride)
    assertWithinRange(rai, offset_pos)
    offset_pos[] = val

else:
  template `[]`*[T](rai: PyArrayRandAccIter[T]): var T =
    (rai.pos[])

  proc `[]=`*[T](rai: PyArrayRandAccIter[T], val: T) {. inline .} =
    rai.pos[] = val

  proc `[]`*[T](rai: PyArrayRandAccIter[T], idx: int): var T =
    let offset_pos = offset_ptr_in_bytes(rai.pos, idx * rai.flatstride)
    return offset_pos[]

  proc `[]=`*[T](rai: PyArrayRandAccIter[T], idx: int, val: T) =
    let offset_pos = offset_ptr_in_bytes(rai.pos, idx * rai.flatstride)
    offset_pos[] = val


proc inc*[T](rai: var PyArrayRandAccIter[T], delta: int) {. inline .} =
  rai.pos = offset_ptr_in_bytes(rai.pos, delta * rai.flatstride)


when doWithinRangeChecks:
  proc inc*[T](rai: var PyArrayRandAccIter[T]) {. inline .} =
    rai.pos = offset_ptr_in_bytes(rai.pos, rai.flatstride)

  proc derefInc*[T](rai: var PyArrayRandAccIter[T]): var T {. inline .} =
    assertWithinRange(rai)
    let prev: ptr T = rai.pos
    rai.pos = offset_ptr(prev)
    result = prev[]

else:
  proc inc*[T](rai: var PyArrayRandAccIter[T]) {. inline .} =
    rai.pos = cast[ptr T](offset_void_ptr_in_bytes(rai.pos, rai.flatstride))

  proc derefInc*[T](rai: var PyArrayRandAccIter[T]): var T {. inline .} =
    let prev: ptr T = rai.pos
    rai.pos = offset_ptr(prev)
    result = prev[]


proc dec*[T](rai: var PyArrayRandAccIter[T], delta: int) {. inline .} =
  rai.pos = offset_ptr_in_bytes(rai.pos, -delta * rai.flatstride)


when doWithinRangeChecks:
  proc dec*[T](rai: var PyArrayRandAccIter[T]) {. inline .} =
    rai.pos = offset_ptr(rai.pos, -1)
else:
  proc dec*[T](rai: var PyArrayRandAccIter[T]) {. inline .} =
    rai.pos = cast[ptr T](offset_void_ptr_in_bytes(rai.pos, -(rai.flatstride)))


proc `+`*[T](rai: PyArrayRandAccIter[T], delta: int):
    PyArrayRandAccIter[T] {. inline .} =
  result.pos = offset_ptr(rai.pos, delta)
  result.arr = rai.arr
  result.flatstride = rai.flatstride
  when doWithinRangeChecks:
    # Check ranges.  Catch mistakes.
    result.low = rai.low
    result.high = rai.high


proc `-`*[T](rai: PyArrayRandAccIter[T], delta: int):
    PyArrayRandAccIter[T] {. inline .} =
  result.pos = offset_ptr(rai.pos, -delta)
  result.arr = rai.arr
  result.flatstride = rai.flatstride
  when doWithinRangeChecks:
    # Check ranges.  Catch mistakes.
    result.low = rai.low
    result.high = rai.high


when doSamePyArrayChecks:
  # Check that our iterators are pointing at the same array.

  template isNotSamePyArray[T](bounds: PyArrayIterBounds[T];
      rai: PyArrayRandAccIter[T]): bool =
    (bounds.arr != rai.arr)

  proc assertSamePyArray[T](bounds: PyArrayIterBounds[T];
      rai: PyArrayRandAccIter[T]) =
    # Note: Use a proc rather than a template, to get a fuller stack trace.
    if isNotSamePyArray(bounds, rai):
      let msg = "A PyArrayRandAccIter[$1] was compared to a PyArrayIterBounds[$1], but they point to different PyArrayObjects" %
          getCompileTimeType(T)
      raise newException(ValueError, msg)

  proc contains*[T](bounds: PyArrayIterBounds[T];
      rai: PyArrayRandAccIter[T]): bool {.inline.} =
    ## Test whether the PyArrayRandAccIter is within its bounds.
    ##
    ## This is intended to be used by user code (in contrast to `isNotWithinRange`,
    ## which is not intended to be used by user code; it is for range-checking
    ## that will be disabled in release builds).
    assertSamePyArray(bounds, rai)
    (bounds.low <= rai.pos and rai.pos <= bounds.high)

  template isNotSamePyArray[T](lhs, rhs: PyArrayRandAccIter[T]): bool =
    (lhs.arr != rhs.arr)

  proc assertSamePyArray[T](lhs, rhs: PyArrayRandAccIter[T]) =
    # Note: Use a proc rather than a template, to get a fuller stack trace.
    if isNotSamePyArray(lhs, rhs):
      let msg = "Two PyArrayRandAccIter[$1] were compared, but they point to different PyArrayObjects" %
          getCompileTimeType(T)
      raise newException(ValueError, msg)

  proc `==`*[T](lhs, rhs: PyArrayRandAccIter[T]): bool {.inline.} =
    ## Note:  If possible, use the (iter in bounds) idiom instead.
    assertSamePyArray(lhs, rhs)
    (lhs.pos == rhs.pos)

  proc `!=`*[T](lhs, rhs: PyArrayRandAccIter[T]): bool {.inline.} =
    ## Note:  If possible, use the (iter in bounds) idiom instead.
    assertSamePyArray(lhs, rhs)
    (lhs.pos != rhs.pos)

  proc `<=`*[T](lhs, rhs: PyArrayRandAccIter[T]): bool {.inline.} =
    ## Note:  If possible, use the (iter in bounds) idiom instead.
    assertSamePyArray(lhs, rhs)
    (cast[int](lhs.pos) <= cast[int](rhs.pos))

  proc `<`*[T](lhs, rhs: PyArrayRandAccIter[T]): bool {.inline.} =
    ## Note:  If possible, use the (iter in bounds) idiom instead.
    assertSamePyArray(lhs, rhs)
    (cast[int](lhs.pos) < cast[int](rhs.pos))

else:

  template contains*[T](bounds: PyArrayIterBounds[T],
      rai: PyArrayRandAccIter[T]): bool =
    ## Test whether the PyArrayRandAccIter is within its bounds.
    ##
    ## This is intended to be used by user code (in contrast to `isNotWithinRange`,
    ## which is not intended to be used by user code; it is for range-checking
    ## that will be disabled in release builds).
    (bounds.low <= rai.pos and rai.pos <= bounds.high)

  template `==`*[T](lhs, rhs: PyArrayRandAccIter[T]): bool =
    ## Note:  If possible, use the (iter in bounds) idiom instead.
    (lhs.pos == rhs.pos)

  template `!=`*[T](lhs, rhs: PyArrayRandAccIter[T]): bool =
    ## Note:  If possible, use the (iter in bounds) idiom instead.
    (lhs.pos != rhs.pos)

  template `<=`*[T](lhs, rhs: PyArrayRandAccIter[T]): bool =
    ## Note:  If possible, use the (iter in bounds) idiom instead.
    (cast[int](lhs.pos) <= cast[int](rhs.pos))

  template `<`*[T](lhs, rhs: PyArrayRandAccIter[T]): bool =
    ## Note:  If possible, use the (iter in bounds) idiom instead.
    (cast[int](lhs.pos) < cast[int](rhs.pos))


proc getBounds*[T](iter: PyArrayForwardIter[T]): PyArrayIterBounds[T] {.inline.} =
  ## Return a PyArrayIterBounds over type `T`.
  result = initPyArrayIterBounds[T](iter.arr)

proc getBounds*[T](iter: PyArrayRandAccIter[T]): PyArrayIterBounds[T] {.inline.} =
  ## Return a PyArrayIterBounds over type `T`.
  result = initPyArrayIterBounds[T](iter.arr)


proc getNumElemsRemaining*[T](iter: PyArrayForwardIter[T]; bounds: PyArrayIterBounds[T]):
    int {.inline.} =
  ## Get the number of distinct elements remaining, that are accessible by
  ## single increments of `iter` within `bounds`.  Return 0 if `iter` is not
  ## within `bounds` (because in this case, any for-loop or while-loop should
  ## exit immediately).
  let hasNotWrapped: bool = cast[int](iter.pos) >= cast[int](bounds.low)
  let numBytes: int = cast[int](bounds.high) - cast[int](iter.pos)
  # if (numBytes == 0), then (iter.pos == bounds.high).
  # So there are no more steps, but 1 more element remaining (ie, this one).
  if numBytes >= 0 and hasNotWrapped:
    let numSteps: int = numBytes div sizeof(T)
    result = numSteps + 1
  else:
    result = 0


proc getNumElemsRemaining*[T](iter: PyArrayRandAccIter[T]; bounds: PyArrayIterBounds[T]):
    int {.inline.} =
  ## Get the number of distinct elements remaining, that are accessible by
  ## single increments of `iter` within `bounds`.  Return 0 if `iter` is not
  ## within `bounds` (because in this case, any for-loop or while-loop should
  ## exit immediately).
  let isNotBeforeLow: bool = cast[int](iter.pos) >= cast[int](bounds.low)
  let numBytes: int = cast[int](bounds.high) - cast[int](iter.pos)
  # if (numBytes == 0), then (iter.pos == bounds.high).
  # So there are no more steps, but 1 more element remaining (ie, this one).
  if numBytes >= 0 and isNotBeforeLow:
    # OK, there's at least 1 element remaining (this one).  But how many more?
    # We must be careful:  The flatstrides of the PyArrayRandAccIter might not
    # fit exactly into the total number of bytes remaining.  (There might be
    # both an initial offset and an increment delta.)
    let strideBytes: int = iter.flatstride
    let numSteps: int = numBytes div strideBytes
    result = numSteps + 1
  else:
    result = 0

