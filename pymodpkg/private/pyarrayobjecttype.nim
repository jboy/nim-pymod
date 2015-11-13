# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

## Nim type definition and accessors for the Numpy C-API PyArrayObject type.
##
## Documentation here:
##  http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArray_Type
##  http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArrayObject
##  http://docs.scipy.org/doc/numpy/reference/c-api.array.html#array-structure-and-data-access
##
## Most of these functions are preceded by the prefix `PyArray_` in the C-API.
##
## Instead, we prefix PyArrayObject accessor & creation functions with `get`
## (in place of the `PyArray_` prefix).  For example: PyArray_NDIM -> getNDIM


import pymodpkg/ptrutils
import pymodpkg/private/nptypes
import pymodpkg/private/pyarraydescrtype


## Nim type definition of the PyArrayObject type:
##  http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArrayObject 
type PyArrayObject* {. importc: "PyArrayObject", header: "pymodpkg/private/numpyarrayobject.h", final .} = object


# http://nim-lang.org/system.html#csize
type npy_intp* = csize


type CArrayProxy*[T] = object
  ## An indexable object that can be passed around in place of a C array.
  ##
  ## If you have a pointer `p` that points to a C array of known length `n`,
  ## this type allows you to pass around your pointer and index it, as if you
  ## were indexing arrays in C -- but now with soft range checking!
  ##
  ## ("Soft range checking" == return 0 rather than raising an exception.)
  ##
  ## NOTE:  This type should only be used when it is meaningful to return 0
  ## rather than raising an exception if the index is out of bounds.
  ##
  ## Fun fact:  Quite some time after I first created this type, I discovered
  ## that the Numpy C-API has an identical, rarely-used type `PyArray_Dims`:
  ##  http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArray_Dims
  ## (where PyArray_Dims == CArrayProxy[npy_intp]), except that `Pyarray_Dims`
  ## is passed as a parameter BY ADDRESS (ie, by pointer) rather than by value.
  ##
  ## TODO:  Should I replace this type with a Nim version of `PyArray_Dims`?
  p: ptr T
  n: cint

proc getPtr*[T](ap: CArrayProxy[T]): ptr T {. inline .} =
  return ap.p

proc getLen*[T](ap: CArrayProxy[T]): cint {. inline .} =
  return ap.n


proc `[]`*[T](ap: CArrayProxy[T], idx: int): T {. inline .} =
  ## NOTE:  This type should only be used when it is meaningful to return 0
  ## rather than raising an exception if the index `idx` is out of bounds.
  if idx < 0:
    return 0.T
  elif idx >= ap.n:
    return 0.T
  else:
    return offset_ptr(ap.p, idx)[]


proc `$`*[T](ap: CArrayProxy[T]): string =
  # Use memory allocation (a `seq[T]` instance) for a string representation.
  #
  # Based upon the usage example at:
  #  http://nim-lang.org/docs/system.html#newSeq,seq[T],Natural
  let num_elems = ap.getLen
  var ss: seq[string]
  newSeq(ss, num_elems)
  for i in 0..<num_elems:
    ss[i] = $(ap[i])
  result = $ss


iterator items*[T](ap: CArrayProxy[T]): T =
  let p: ptr T = ap.getPtr
  let n: cint = ap.getLen
  var i: cint = 0
  while i < n:
    yield (offset_ptr(p, i)[])
    inc(i)


# These procs mimic the Array C API data-access macros:
#  http://docs.scipy.org/doc/numpy/reference/c-api.array.html#array-structure-and-data-access
# (but with the prefix "PyArray_" changed to the verb "get")

proc getNDIM*(arr: ptr PyArrayObject): cint {.
    importc: "PyArray_NDIM", header: "pymodpkg/private/numpyarrayobject.h", cdecl .}

proc getDIMS*(arr: ptr PyArrayObject): ptr npy_intp {.
    importc: "PyArray_DIMS", header: "pymodpkg/private/numpyarrayobject.h", cdecl .}

proc getSHAPE*(arr: ptr PyArrayObject): ptr npy_intp {.
    importc: "PyArray_SHAPE", header: "pymodpkg/private/numpyarrayobject.h", cdecl .}

proc getDATA*(arr: ptr PyArrayObject): pointer {.
    importc: "PyArray_DATA", header: "pymodpkg/private/numpyarrayobject.h", cdecl .}

proc getSTRIDES*(arr: ptr PyArrayObject): ptr npy_intp {.
    importc: "PyArray_STRIDES", header: "pymodpkg/private/numpyarrayobject.h", cdecl .}

proc getDIM*(arr: ptr PyArrayObject, n: cint): npy_intp {.
    importc: "PyArray_DIM", header: "pymodpkg/private/numpyarrayobject.h", cdecl .}

proc getSTRIDE*(arr: ptr PyArrayObject, n: cint): npy_intp {.
    importc: "PyArray_STRIDE", header: "pymodpkg/private/numpyarrayobject.h", cdecl .}

proc getDESCR*(arr: ptr PyArrayObject): ptr PyArrayDescr {.
    importc: "PyArray_DESCR", header: "pymodpkg/private/numpyarrayobject.h", cdecl .}

proc getDTYPE*(arr: ptr PyArrayObject): ptr PyArrayDescr {.
    importc: "PyArray_DTYPE", header: "pymodpkg/private/numpyarrayobject.h", cdecl .}

proc getFLAGS*(arr: ptr PyArrayObject): cint {.
    importc: "PyArray_FLAGS", header: "pymodpkg/private/numpyarrayobject.h", cdecl .}

proc getITEMSIZE*(arr: ptr PyArrayObject): cint {.
    importc: "PyArray_ITEMSIZE", header: "pymodpkg/private/numpyarrayobject.h", cdecl .}

proc getTYPE*(arr: ptr PyArrayObject): cint {.
    importc: "PyArray_TYPE", header: "pymodpkg/private/numpyarrayobject.h", cdecl .}

proc getPtr*(arr: ptr PyArrayObject, idxes: ptr npy_intp): pointer {.
    importc: "PyArray_NBYTES", header: "pymodpkg/private/numpyarrayobject.h", cdecl .}


# Convenient Nim-style attributes & iterators.
# They are intended to mimic the behaviour of the corresponding attributes
# of the PyArrayObject struct:
#  http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArrayObject

template data*(arr: ptr PyArrayObject): pointer =
  getDATA(arr)

template data*(arr: ptr PyArrayObject; T: typedesc): expr =
  var p: ptr T = cast[ptr T](getDATA(arr))
  p

# The attribute of the PyArrayObject struct.
template nd*(arr: ptr PyArrayObject): cint =
  getNDIM(arr)

# The more-familiar name for the `nd` attribute in Python.
template ndim*(arr: ptr PyArrayObject): cint =
  getNDIM(arr)


iterator enumerateDimensions*(arr: ptr PyArrayObject): tuple[idx: cint, val: npy_intp] {. inline .} =
  ## An iterator for your for-loops.
  let nd: cint = arr.nd
  let p: ptr npy_intp = getDIMS(arr)
  var i: cint = 0
  while i < nd:
    yield (i, offset_ptr(p, i)[])
    inc(i)


proc elcount*(arr: ptr PyArrayObject): npy_intp {. inline .} =
  ## Count the number of elements in the array.
  result = 1
  for i, n in arr.enumerateDimensions:
    result *= n


proc dimensions*(arr: ptr PyArrayObject): CArrayProxy[npy_intp] {. inline .} =
  result.n = arr.nd
  result.p = getDIMS(arr)


proc shape*(arr: ptr PyArrayObject): CArrayProxy[npy_intp] {. inline .} =
  ## NOTE:  There is not actually a `shape` attribute in PyArrayObject, but
  ## since there's already `PyArray_SHAPE` as a synonym for `PyArray_DIMS`,
  ## and the Python object attribute is `shape`, I figure we might as well
  ## offer a `shape` attribute here too.
  result.n = arr.nd
  result.p = getSHAPE(arr)


iterator enumerateStrides*(arr: ptr PyArrayObject): tuple[idx: cint, val: npy_intp] {. inline .} =
  ## An iterator for your for-loops.
  let nd: cint = arr.nd
  let p: ptr npy_intp = getSTRIDES(arr)
  var i: cint = 0
  while i < nd:
    yield (i, offset_ptr(p, i)[])
    inc(i)


proc strides*(arr: ptr PyArrayObject): CArrayProxy[npy_intp] {. inline .} =
  result.n = arr.nd
  result.p = getSTRIDES(arr)


template descr*(arr: ptr PyArrayObject): ptr PyArrayDescr =
  getDESCR(arr)


template dtype*(arr: ptr PyArrayObject): NpType =
  ## This is a convenience alias to enable something that behaves like the
  ## familiar `.dtype` attribute of a Numpy array in Python.  It doesn't
  ## correspond to any attribute of the PyArrayObject struct in C.
  ##
  ## In particular, don't confuse this alias with the `PyArray_DTYPE` access
  ## macro, which is simply a synonym for the `PyArray_DESCR` access macro.
  getDESCR(arr).type_num.toNpType

