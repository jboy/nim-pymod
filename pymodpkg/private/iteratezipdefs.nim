# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import pymodpkg/private/pyarrayiters


proc getBounds(slic: Slice[int]): Slice[int] {.inline.} = slic
  ## An overload to match the overloads for the PyArrayIter types.

proc getNumElemsRemaining(val: int; slic: Slice[int]):
    int {.inline.} =
  ## An overload to match the overloads for the PyArrayIter types.
  let isNotBeforeLow: bool = val >= slic.a
  let numSteps = slic.b - val
  if numSteps >= 0 and isNotBeforeLow:
    result = numSteps + 1
  else:
    result = 0


proc getBeginIter[T](iter: PyArrayForwardIter[T]): PyArrayForwardIter[T] {.inline.} = iter
proc getBeginIter[T](iter: PyArrayRandAccIter[T]): PyArrayRandAccIter[T] {.inline.} = iter
proc getBeginIter(slic: Slice[int]): int {.inline.} = slic.a


template iterateZipImpl1*(iterable1: typed): expr =
  let bounds1 = getBounds(iterable1)
  var iter1 = getBeginIter(iterable1)
  while iter1 in bounds1:
    yield iter1
    inc(iter1)

iterator iterateZip*[T](iterable1: PyArrayForwardIter[T]): PyArrayForwardIter[T] {.inline.} =
  iterateZipImpl1(iterable1)

iterator iterateZip*[T](iterable1: PyArrayRandAccIter[T]): PyArrayRandAccIter[T] {.inline.} =
  iterateZipImpl1(iterable1)

iterator iterateZip*(iterable1: Slice[int]): int {.inline.} =
  iterateZipImpl1(iterable1)


template iterateZipImpl2*(iterable1, iterable2: typed): expr =
  let
    bounds1 = getBounds(iterable1)
    bounds2 = getBounds(iterable2)
  var
    iter1 = getBeginIter(iterable1)
    iter2 = getBeginIter(iterable2)
  let
    numElems1: int = getNumElemsRemaining(iter1, bounds1)
    numElems2: int = getNumElemsRemaining(iter2, bounds2)
  let minNumElems: int = min([numElems1, numElems2])

  if minNumElems == numElems1:
    while iter1 in bounds1:
      yield (iter1, iter2)
      inc(iter1)
      inc(iter2)
  else:
    while iter2 in bounds2:
      yield (iter1, iter2)
      inc(iter1)
      inc(iter2)

iterator iterateZip*[T,I2](iterable1: PyArrayForwardIter[T];
    iterable2: I2):
    (PyArrayForwardIter[T], I2) {.inline.} =
  iterateZipImpl2(iterable1, iterable2)

iterator iterateZip*[T,I2](iterable1: PyArrayRandAccIter[T];
    iterable2: I2):
    (PyArrayRandAccIter[T], I2) {.inline.} =
  iterateZipImpl2(iterable1, iterable2)

iterator iterateZip*[I2](iterable1: Slice[int];
    iterable2: I2):
    (int, I2) {.inline.} =
  iterateZipImpl2(iterable1, iterable2)


template iterateZipImpl3*(iterable1, iterable2, iterable3: typed): expr =
  let
    bounds1 = getBounds(iterable1)
    bounds2 = getBounds(iterable2)
    bounds3 = getBounds(iterable3)
  var
    iter1 = getBeginIter(iterable1)
    iter2 = getBeginIter(iterable2)
    iter3 = getBeginIter(iterable3)
  let
    numElems1: int = getNumElemsRemaining(iter1, bounds1)
    numElems2: int = getNumElemsRemaining(iter2, bounds2)
    numElems3: int = getNumElemsRemaining(iter3, bounds3)
  let minNumElems: int = min([numElems1, numElems2, numElems3])

  if minNumElems == numElems1:
    while iter1 in bounds1:
      yield (iter1, iter2, iter3)
      inc(iter1)
      inc(iter2)
      inc(iter3)
  elif minNumElems == numElems2:
    while iter2 in bounds2:
      yield (iter1, iter2, iter3)
      inc(iter1)
      inc(iter2)
      inc(iter3)
  else:
    while iter3 in bounds3:
      yield (iter1, iter2, iter3)
      inc(iter1)
      inc(iter2)
      inc(iter3)

iterator iterateZip*[T,I2,I3](iterable1: PyArrayForwardIter[T];
    iterable2: I2; iterable3: I3):
    (PyArrayForwardIter[T], I2, I3) {.inline.} =
  iterateZipImpl3(iterable1, iterable2, iterable3)

iterator iterateZip*[T,I2,I3](iterable1: PyArrayRandAccIter[T];
    iterable2: I2; iterable3: I3):
    (PyArrayRandAccIter[T], I2, I3) {.inline.} =
  iterateZipImpl3(iterable1, iterable2, iterable3)

iterator iterateZip*[I2,I3](iterable1: Slice[int];
    iterable2: I2; iterable3: I3):
    (int, I2, I3) {.inline.} =
  iterateZipImpl3(iterable1, iterable2, iterable3)


template iterateZipImpl4*(iterable1, iterable2, iterable3, iterable4: typed): expr =
  let
    bounds1 = getBounds(iterable1)
    bounds2 = getBounds(iterable2)
    bounds3 = getBounds(iterable3)
    bounds4 = getBounds(iterable4)
  var
    iter1 = getBeginIter(iterable1)
    iter2 = getBeginIter(iterable2)
    iter3 = getBeginIter(iterable3)
    iter4 = getBeginIter(iterable4)
  let
    numElems1: int = getNumElemsRemaining(iter1, bounds1)
    numElems2: int = getNumElemsRemaining(iter2, bounds2)
    numElems3: int = getNumElemsRemaining(iter3, bounds3)
    numElems4: int = getNumElemsRemaining(iter4, bounds4)
  let minNumElems: int = min([numElems1, numElems2, numElems3, numElems4])

  if minNumElems == numElems1:
    while iter1 in bounds1:
      yield (iter1, iter2, iter3, iter4)
      inc(iter1)
      inc(iter2)
      inc(iter3)
      inc(iter4)
  elif minNumElems == numElems2:
    while iter2 in bounds2:
      yield (iter1, iter2, iter3, iter4)
      inc(iter1)
      inc(iter2)
      inc(iter3)
      inc(iter4)
  elif minNumElems == numElems3:
    while iter3 in bounds3:
      yield (iter1, iter2, iter3, iter4)
      inc(iter1)
      inc(iter2)
      inc(iter3)
      inc(iter4)
  else:
    while iter4 in bounds4:
      yield (iter1, iter2, iter3, iter4)
      inc(iter1)
      inc(iter2)
      inc(iter3)
      inc(iter4)

iterator iterateZip*[T,I2,I3,I4](iterable1: PyArrayForwardIter[T];
    iterable2: I2; iterable3: I3; iterable4: I4):
    (PyArrayForwardIter[T], I2, I3, I4) {.inline.} =
  iterateZipImpl4(iterable1, iterable2, iterable3, iterable4)

iterator iterateZip*[T,I2,I3,I4](iterable1: PyArrayRandAccIter[T];
    iterable2: I2; iterable3: I3; iterable4: I4):
    (PyArrayRandAccIter[T], I2, I3, I4) {.inline.} =
  iterateZipImpl4(iterable1, iterable2, iterable3, iterable4)

iterator iterateZip*[I2,I3,I4](iterable1: Slice[int];
    iterable2: I2; iterable3: I3; iterable4: I4):
    (int, I2, I3, I4) {.inline.} =
  iterateZipImpl4(iterable1, iterable2, iterable3, iterable4)


template iterateZipImpl5*(iterable1, iterable2, iterable3, iterable4, iterable5: typed): expr =
  let
    bounds1 = getBounds(iterable1)
    bounds2 = getBounds(iterable2)
    bounds3 = getBounds(iterable3)
    bounds4 = getBounds(iterable4)
    bounds5 = getBounds(iterable5)
  var
    iter1 = getBeginIter(iterable1)
    iter2 = getBeginIter(iterable2)
    iter3 = getBeginIter(iterable3)
    iter4 = getBeginIter(iterable4)
    iter5 = getBeginIter(iterable5)
  let
    numElems1: int = getNumElemsRemaining(iter1, bounds1)
    numElems2: int = getNumElemsRemaining(iter2, bounds2)
    numElems3: int = getNumElemsRemaining(iter3, bounds3)
    numElems4: int = getNumElemsRemaining(iter4, bounds4)
    numElems5: int = getNumElemsRemaining(iter5, bounds5)
  let minNumElems: int = min([numElems1, numElems2, numElems3, numElems4, numElems5])

  if minNumElems == numElems1:
    while iter1 in bounds1:
      yield (iter1, iter2, iter3, iter4, iter5)
      inc(iter1)
      inc(iter2)
      inc(iter3)
      inc(iter4)
      inc(iter5)
  elif minNumElems == numElems2:
    while iter2 in bounds2:
      yield (iter1, iter2, iter3, iter4, iter5)
      inc(iter1)
      inc(iter2)
      inc(iter3)
      inc(iter4)
      inc(iter5)
  elif minNumElems == numElems3:
    while iter3 in bounds3:
      yield (iter1, iter2, iter3, iter4, iter5)
      inc(iter1)
      inc(iter2)
      inc(iter3)
      inc(iter4)
      inc(iter5)
  elif minNumElems == numElems4:
    while iter4 in bounds4:
      yield (iter1, iter2, iter3, iter4, iter5)
      inc(iter1)
      inc(iter2)
      inc(iter3)
      inc(iter4)
      inc(iter5)
  else:
    while iter5 in bounds5:
      yield (iter1, iter2, iter3, iter4, iter5)
      inc(iter1)
      inc(iter2)
      inc(iter3)
      inc(iter4)
      inc(iter5)

iterator iterateZip*[T,I2,I3,I4,I5](iterable1: PyArrayForwardIter[T];
    iterable2: I2; iterable3: I3; iterable4: I4; iterable5: I5):
    (PyArrayForwardIter[T], I2, I3, I4, I5) {.inline.} =
  iterateZipImpl5(iterable1, iterable2, iterable3, iterable4, iterable5)

iterator iterateZip*[T,I2,I3,I4,I5](iterable1: PyArrayRandAccIter[T];
    iterable2: I2; iterable3: I3; iterable4: I4; iterable5: I5):
    (PyArrayRandAccIter[T], I2, I3, I4, I5) {.inline.} =
  iterateZipImpl5(iterable1, iterable2, iterable3, iterable4, iterable5)

iterator iterateZip*[I2,I3,I4,I5](iterable1: Slice[int];
    iterable2: I2; iterable3: I3; iterable4: I4; iterable5: I5):
    (int, I2, I3, I4, I5) {.inline.} =
  iterateZipImpl5(iterable1, iterable2, iterable3, iterable4, iterable5)

