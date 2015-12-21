{. compile: "pymodpkg/private/pyarrayobject_c.c" .}

# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

## Nim wrappers for functions in the Numpy C-API Array API.
##
## Numpy C-API Array API documentation here:
##  http://docs.scipy.org/doc/numpy/reference/c-api.html
##  http://docs.scipy.org/doc/numpy/reference/c-api.array.html
##  http://docs.scipy.org/doc/numpy/reference/c-api.config.html
##  http://docs.scipy.org/doc/numpy/reference/c-api.dtype.html
##  http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html
##
## Most of these functions are preceded by the prefix `PyArray_` in the C-API.
##
## Instead, we prefix the PyArrayObject attribute accessor functions with `get`
## (in place of the `PyArray_` prefix).  For example:
##  - PyArray_NDIM -> getNDIM
##  - PyArray_SHAPE -> getSHAPE
##  - etc.
##
## We prefix PyArrayObject creation functions with `create` (in place of the
## `PyArray_` prefix).  These functions allocate new PyArrayObject instances.
## For example:
##  - PyArray_SimpleNew -> createSimpleNew
##  - PyArray_NewLikeArray -> createNewLikeArray
##
## Additionally, if a PyArrayObject creation function allocates new memory for
## a copy of the underlying data, we suffix the function name with `NewData`.
## For example:
##  - PyArray_NewCopy -> createNewCopyNewData
##
## If a PyArrayObject creation function _might_ allocate new memory for a copy
## of the data, we suffix the function name with `MaybeNewData`.  Rather than
## just returning a `ptr PyArrayObject`, a function like this will return a
## tuple[newArray: ptr PyArrayObject, hasNewData: bool].  For example:
##  - PyArray_NewShape -> createNewShapeMaybeNewData
##
## For functions like `createNewShapeMaybeNewData` that might allocate a copy
## of the underlying data (or might not), we also provide an alternate version
## `createNewShapeNewData` that _always_ allocates a new copy of the underlying
## data.  This alternate version simply returns a `ptr PyArrayObject`.
##
## Finally, if a function doesn't return anything, we prefix its name with `do`.
## For example:
##  - PyArray_CopyInto -> doCopyInto
##  - PyArray_FILLWBYTE -> doFILLWBYTE
##
## The Numpy C functions to create new arrays require both a shape parameter &
## a dtype parameter.
##
## In the Numpy C-API, an array's shape is described using the attribute pair
## `(nd: cint, dims: ptr npy_intp)`.  This pair of attributes is also supplied
## to Numpy C-API array creation functions to specify the shape of a new array.
## Pymod provides the wrapper procs `arr.getNDIM()` & `arr.getDIMS()` to access
## these attributes.
##
## However, it is not necessary to use these two wrapper procs; Pymod makes it
## easier to specify a shape parameter.  In Pymod, there are 2 ways to specify
## a shape parameter:
##  1. `CArrayProxy[npy_intp]`, a simple Nim wrapper type around
##     `(nd: cint, dims: ptr npy_intp)`.  A `CArrayProxy[npy_intp]` instance
##     is returned by the PyArrayObject accessors `.dimensions`, `strides` &
##     `.shape`.
##  2. `openArray[int]`, so you can supply Nim array literals; eg, `[5, 4, 3]`.
##
## There are 4 ways to specify a dtype parameter:
##  1, `dtype: ptr PyArrayDescr`, as per the Numpy C-API.
##  2. `typenum: cint`, as per the Numpy C-API.
##  3. `typenum: CNpyTypes`, a Nim enum that matches the C-API enum `NPY_TYPES`
##     in `<numpy/ndarraytypes.h>`; eg: `CNpyTypes.NPY_UBYTE`.
##  4. `nptype: NpType`, a Nim enum that contains *only* the dtypes that Pymod
##     supports; eg: `np_uint8`.  This is the recommended method in Pymod.
##
## Any proc that either allocates a new PyObject that must be memory-managed
## (such as a PyArrayObject), or could raise an exception, will be provided as
## a template wrapper around an implementation proc.  This enables us to use
## the template's instantiation info in debugging info and exception traces.
##
## Here's the full list of procs & templates:
##
##  - getDescrFromType(typenum: cint): ptr PyArrayDescr
##  - getDescrFromType(typenum: CNpyTypes): ptr PyArrayDescr
##  - getDescrFromType(nptype: NpType): ptr PyArrayDescr
##
##  - canCastArrayTo(arr: ptr PyArrayObject, toType: ptr PyArray_Descr,
##        casting: NpyCastingAlternatives): bool
##
##  - createNewLikeArray(prototype: ptr PyArrayObject, order: NpyOrderAlternatives,
##        dtype: ptr PyArrayDescr, subOk: bool): ptr PyArrayObject
##
##  - createSimpleNew(dims: CArrayProxy[npy_intp], nptype: NpType): ptr PyArrayObject
##  - createSimpleNew(dims: openarray[int], nptype: NpType): ptr PyArrayObject
##
##  - createNewCopyNewData(old: ptr PyArrayObject, order: NpyOrderAlternatives): ptr PyArrayObject
##  - copy(old: ptr PyArrayObject): ptr PyArrayObject  # an alias for `createNewCopyNewData`
##  - createAsTypeNewData(old: ptr PyArrayObject, newtype: NpType): ptr PyArrayObject
##
##  - doCopyInto(dest: ptr PyArrayObject, src: ptr PyArrayObject)
##  - doFILLWBYTE(arr: ptr PyArrayObject, val: cint)
##
##  - doResizeDataInplace(old: ptr PyArrayObject, newShape: openarray[int], doRefCheck: bool=true)
##  - doResizeDataInplaceNumRows(old: ptr PyArrayObject, newNumRows: int, doRefCheck: bool=true)


## New procs to add:
##
##  - createNewShapeMaybeNewData(old: ptr PyArrayObject, newShape: openarray[int],
##        order: NpyOrderAlternatives): tuple[newArray: ptr PyArrayObject, hasNewData: bool]
##  - createNewShapeNewData(old: ptr PyArrayObject, newShape: openarray[int],
##        order: NpyOrderAlternatives): ptr PyArrayObject

##  - createSwapAxes(old: ptr PyArrayObject, axis1: cint, axis2: cint): ptr PyArrayObject
##  - createTranspose(old: ptr PyArrayObject): ptr PyArrayObject
##  - createTranspose(old: ptr PyArrayObject, permute: openarray[int]): ptr PyArrayObject
##  - createFlattenNewData(arr: ptr PyArrayObject,
##        order: NpyOrderAlternatives): ptr PyArrayObject
##  - createRavelMaybeNewData(arr: ptr PyArrayObject,
##        order: NpyOrderAlternatives): tuple[newArray: ptr PyArrayObject, hasNewData: bool]

##  - createEmpty(nd: cint, dims: ptr npy_intp, dtype: ptr PyArrayDescr,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createEmpty(nd: cint, dims: ptr npy_intp, typenum: cint,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createEmpty(nd: cint, dims: ptr npy_intp, typenum: CNpyTypes,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createEmpty(nd: cint, dims: ptr npy_intp, nptype: NpType,
##        isFortranOrder: bool=false): ptr PyArrayObject
##
##  - createEmpty(dims: CArrayProxy[npy_intp], dtype: ptr PyArrayDescr,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createEmpty(dims: CArrayProxy[npy_intp], typenum: cint,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createEmpty(dims: CArrayProxy[npy_intp], typenum: CNpyTypes,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createEmpty(dims: CArrayProxy[npy_intp], nptype: NpType,
##        isFortranOrder: bool=false): ptr PyArrayObject
##
##  - createEmpty(dims: openarray[int], dtype: ptr PyArrayDescr,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createEmpty(dims: openarray[int], typenum: cint,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createEmpty(dims: openarray[int], typenum: CNpyTypes,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createEmpty(dims: openarray[int], nptype: NpType,
##        isFortranOrder: bool=false): ptr PyArrayObject
##
##  - createZeros(nd: cint, dims: ptr npy_intp, dtype: ptr PyArrayDescr,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createZeros(nd: cint, dims: ptr npy_intp, typenum: cint,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createZeros(nd: cint, dims: ptr npy_intp, typenum: CNpyTypes,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createZeros(nd: cint, dims: ptr npy_intp, nptype: NpType,
##        isFortranOrder: bool=false): ptr PyArrayObject
##
##  - createZeros(dims: CArrayProxy[npy_intp], dtype: ptr PyArrayDescr,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createZeros(dims: CArrayProxy[npy_intp], typenum: cint,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createZeros(dims: CArrayProxy[npy_intp], typenum: CNpyTypes,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createZeros(dims: CArrayProxy[npy_intp], nptype: NpType,
##        isFortranOrder: bool=false): ptr PyArrayObject
##
##  - createZeros(dims: openarray[int], dtype: ptr PyArrayDescr,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createZeros(dims: openarray[int], typenum: cint,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createZeros(dims: openarray[int], typenum: CNpyTypes,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createZeros(dims: openarray[int], nptype: NpType,
##        isFortranOrder: bool=false): ptr PyArrayObject
##
##  - createFull(nd: cint, dims: ptr npy_intp, fillValue: double, dtype: ptr PyArrayDescr,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createFull(nd: cint, dims: ptr npy_intp, fillValue: double, typenum: cint,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createFull(nd: cint, dims: ptr npy_intp, fillValue: double, typenum: CNpyTypes,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createFull(nd: cint, dims: ptr npy_intp, fillValue: double, nptype: NpType,
##        isFortranOrder: bool=false): ptr PyArrayObject
##
##  - createFull(nd: cint, dims: CArrayProxy[npy_intp], fillValue: double, dtype: ptr PyArrayDescr,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createFull(nd: cint, dims: CArrayProxy[npy_intp], fillValue: double, typenum: cint,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createFull(nd: cint, dims: CArrayProxy[npy_intp], fillValue: double, typenum: CNpyTypes,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createFull(nd: cint, dims: CArrayProxy[npy_intp], fillValue: double, nptype: NpType,
##        isFortranOrder: bool=false): ptr PyArrayObject
##
##  - createFull(dims: openarray[int], fillValue: double, dtype: ptr PyArrayDescr,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createFull(dims: openarray[int], fillValue: double, typenum: cint,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createFull(dims: openarray[int], fillValue: double, typenum: CNpyTypes,
##        isFortranOrder: bool=false): ptr PyArrayObject
##  - createFull(dims: openarray[int], fillValue: double, nptype: NpType,
##        isFortranOrder: bool=false): ptr PyArrayObject

##  - createEmptyLike(prototype: ptr PyArrayObject,
##        dtype: ptr PyArrayDescr=nil,
##        order: NpyOrderAlternatives=NpyOrderAlternatives.keeporder,
##        subOk: bool=true):
##            ptr PyArrayObject
##
##  - createZerosLike(prototype: ptr PyArrayObject,
##        dtype: ptr PyArrayDescr=nil,
##        order: NpyOrderAlternatives=NpyOrderAlternatives.keeporder,
##        subOk: bool=true):
##            ptr PyArrayObject
##
##  - createFullLike(prototype: ptr PyArrayObject,
##        fillValue: double,
##        dtype: ptr PyArrayDescr=nil,
##        order: NpyOrderAlternatives=NpyOrderAlternatives.keeporder,
##        subOk: bool=true):
##            ptr PyArrayObject


import strutils
import typetraits  # name(t: typedesc)

import pymodpkg/miscutils
import pymodpkg/pyobject

import pymodpkg/private/membrain

import pymodpkg/private/nptypes
export nptypes.NumpyCompatibleNimType
export nptypes.NpType
export nptypes.toNpType

import pymodpkg/private/cnpytypes
export cnpytypes.CNpyTypes
export cnpytypes.toNpType
export cnpytypes.toCNpyTypes

import pymodpkg/private/pyarraydescrtype
export pyarraydescrtype.PyArrayDescr
export pyarraydescrtype.dkind
export pyarraydescrtype.dtype
export pyarraydescrtype.byteorder
export pyarraydescrtype.dflags
export pyarraydescrtype.type_num
export pyarraydescrtype.elsize
export pyarraydescrtype.alignment

import pymodpkg/private/pyarrayenums
export pyarrayenums.NpyOrderAlternatives
export pyarrayenums.NpyCastingAlternatives

import pymodpkg/private/pyarrayflags
export pyarrayflags.NPY_ARRAY_BEHAVED
export pyarrayflags.NPY_ARRAY_CARRAY
export pyarrayflags.NPY_ARRAY_CARRAY_RO
export pyarrayflags.PyArrayFlags
export pyarrayflags.flags

import pymodpkg/private/pyarrayobjecttype
export pyarrayobjecttype.PyArrayObject
export pyarrayobjecttype.npy_intp
export pyarrayobjecttype.CArrayProxy
export pyarrayobjecttype.getLen
export pyarrayobjecttype.getPtr
export pyarrayobjecttype.items
export pyarrayobjecttype.`[]`
export pyarrayobjecttype.`$`

export pyarrayobjecttype.getNDIM
export pyarrayobjecttype.getDIMS
export pyarrayobjecttype.getSHAPE
export pyarrayobjecttype.getDATA
export pyarrayobjecttype.getSTRIDES
export pyarrayobjecttype.getDIM
export pyarrayobjecttype.getSTRIDE
export pyarrayobjecttype.getDESCR
export pyarrayobjecttype.getDTYPE
export pyarrayobjecttype.getFLAGS
export pyarrayobjecttype.getITEMSIZE
export pyarrayobjecttype.getTYPE
export pyarrayobjecttype.getPtr

export pyarrayobjecttype.data
export pyarrayobjecttype.nd
export pyarrayobjecttype.ndim
export pyarrayobjecttype.enumerateDimensions
export pyarrayobjecttype.dimensions
export pyarrayobjecttype.elcount
export pyarrayobjecttype.shape
export pyarrayobjecttype.enumerateStrides
export pyarrayobjecttype.strides
export pyarrayobjecttype.descr
export pyarrayobjecttype.dtype

import pymodpkg/private/pyarrayiters
export pyarrayiters.PyArrayForwardIter
export pyarrayiters.`[]`
export pyarrayiters.`[]=`
export pyarrayiters.inc
export pyarrayiters.derefInc
export pyarrayiters.incFast
export pyarrayiters.PyArrayRandAccIter
export pyarrayiters.PyArrayIterBounds
export pyarrayiters.contains
export pyarrayiters.dec
export pyarrayiters.`+`
export pyarrayiters.`-`
export pyarrayiters.`==`
export pyarrayiters.`!=`
export pyarrayiters.`<=`
export pyarrayiters.`<`


## A convenient and plausible maximum number of dimensions to support.
## (This is Numpy's internal limit.)
const NPY_MAXDIMS = 32


## Allow implicit conversion from `ptr PyArrayObject` to `ptr PyObject`
## (since we have that pesky "strong static typing" thing in Nim).
##  http://nim-lang.org/manual.html#converters
##  http://nim-lang.org/manual.html#convertible-relation
converter toPyObject*(obj: ptr PyArrayObject): ptr PyObject = cast[ptr PyObject](obj)
converter toPyObject*(obj: ptr PyArrayDescr): ptr PyObject = cast[ptr PyObject](obj)


# http://nim-lang.org/system.html#instantiationInfo,
type InstantiationInfoTuple = tuple[filename: string, line: int]


proc assertArrayType*(obj: ptr PyArrayObject, NT: typedesc[NumpyCompatibleNimType],
    ii: InstantiationInfoTuple, procname: string) =
  let obj_dtype = obj.dtype  # This requires a lookup in a case statement.
  if toNpType(NT) != obj_dtype:
    let msg = "$1: PyArrayObject supplied dtype `$2` does not match specified Nim type `$3` [File \"$4\", line $5]" %
        [procname, $obj_dtype, NT.name, ii.filename, $ii.line]
    raise newException(ObjectConversionError, msg)


proc assertArrayNdim*(obj: ptr PyArrayObject, expected_nd: Positive,
    ii: InstantiationInfoTuple, procname: string) =
  let obj_nd = obj.nd
  if expected_nd != obj_nd:
    let msg = "$1: PyArrayObject supplied ndim (== $2) does not match specified ndim (== $3) [File \"$4\", line $5]" %
        [procname, $obj_nd, $expected_nd, ii.filename, $ii.line]
    raise newException(ObjectConversionError, msg)


proc assertArrayCContigForIterator*(obj: ptr PyArrayObject,
    ii: InstantiationInfoTuple, procname: string) =
  let is_c_contig: bool = flagBitIsOn(getFLAGS(obj), c_contiguous)
  if not is_c_contig:
    let msg = "$1: PyArrayObject iterator can only be used with C-contiguous data [File \"$2\", line $3]" %
        [procname, ii.filename, $ii.line]
    raise newException(AssertionError, msg)


proc iterateFlatImpl(arr: ptr PyArrayObject, NimT: typedesc[NumpyCompatibleNimType],
    ii: InstantiationInfoTuple, procname: string{lit}):
    PyArrayForwardIter[NimT] =
  assertArrayType(arr, NimT, ii, procname)
  assertArrayCContigForIterator(arr, ii, procname)
  result = initPyArrayForwardIter[NimT](arr)


template iterateFlat*(arr: ptr PyArrayObject, NimT: typedesc[NumpyCompatibleNimType]):
    PyArrayForwardIter[NimT] =
  ## Return a PyArrayForwardIter over type `NimT`.
  ##
  ## A PyArrayForwardIter is an iterator that can only step forward in
  ## single increments.  That is, it can only be incremented and dereferenced.
  ##
  ## Due to the limited manner in which the PyArrayForwardIter's position
  ## can be changed, a PyArrayForwardIter is slightly more predictable than
  ## a PyArrayRandAccIter.  So prefer to use PyArrayForwardIter
  ## if you can.
  ##
  ## NOTE:  This proc requires that the PyArrayObject data is C-contiguous;
  ## else, an AssertionError will be raised.
  ##
  ## Here's an example of how you use this type of iterator:
  ##
  ##   let dt = arr.dtype
  ##   if dt == np_int32:
  ##       let bounds = arr.getBounds(int32)
  ##       var iter = arr.iterateFlat(int32)
  ##       while iter in bounds:
  ##           doSomethingWith(iter[])
  ##           inc(iter)
  ##

  # http://nim-lang.org/system.html#instantiationInfo,
  let ii = instantiationInfo()
  iterateFlatImpl(arr, NimT, ii, "iterateFlat")


iterator items*[T](iter: PyArrayForwardIter[T]):
    PyArrayForwardIter[T] {.inline.} =
  let bounds = iter.getBounds()
  var iter = iter
  while iter in bounds:
    yield iter
    inc(iter)

iterator iterateFlatFast*[T](arr: ptr PyArrayObject; NimT: typedesc[NumpyCompatibleNimType];
    positiveDelta: Positive): PyArrayForwardIter[T] {.inline.} =
  ## "Fast forward"
  # http://nim-lang.org/system.html#instantiationInfo,
  let ii = instantiationInfo()
  let bounds = arr.getBounds()
  var iter = arr.iterateFlatImpl(NimT, ii, "iterateFlatFast")
  while iter in bounds:
    yield iter
    incFast(iter, positiveDelta)

iterator iterateFlatFast*[T](arr: ptr PyArrayObject; NimT: typedesc[NumpyCompatibleNimType];
    positiveOffset, positiveDelta: Positive): PyArrayForwardIter[T] {.inline.} =
  ## "Fast forward"
  # http://nim-lang.org/system.html#instantiationInfo,
  let ii = instantiationInfo()
  let bounds = arr.getBounds()
  var iter = arr.iterateFlatImpl(NimT, ii, "iterateFlatFast")
  incFast(iter, positiveOffset)
  while iter in bounds:
    yield iter
    incFast(iter, positiveDelta)

iterator values*(arr: ptr PyArrayObject, NimT: typedesc[NumpyCompatibleNimType]):
    NimT {.inline.} =
  let bounds = arr.getBounds(NimT)
  var iter = arr.iterateFlat(NimT)
  while iter in bounds:
    yield iter[]
    inc(iter)


proc accessFlatImpl(arr: ptr PyArrayObject; NimT: typedesc[NumpyCompatibleNimType];
    ii: InstantiationInfoTuple; procname: string{lit}; initOffset, incDelta: int):
    PyArrayRandAccIter[NimT] =
  assertArrayType(arr, NimT, ii, procname)
  assertArrayCContigForIterator(arr, ii, procname)
  result = initPyArrayRandAccIter[NimT](arr, initOffset, incDelta)


template accessFlat*(arr: ptr PyArrayObject, NimT: typedesc[NumpyCompatibleNimType]):
    PyArrayRandAccIter[NimT] =
  ## Return a PyArrayRandAccIter over type `NimT`.
  ##
  ## A PyArrayRandAccIter is an iterator that allows random access to
  ## the array's data as a flat 1-D array.  Thus, a PyArrayRandAccIter
  ## is effectively a pointer with range checking.  The iterator position can be
  ## shifted forward or backward by any arbitrary integer amount; in addition,
  ## the iterator can be dereferenced _or_ it can be indexed by any arbitrary
  ## integer amount.
  ##
  ## When the code has been compiled in non-release mode, the iterator position
  ## is range-checked against the iterator's internal bounds when the iterator
  ## is dereferenced or indexed.  Thus, the iterator position can be shifted
  ## arbitrarily outside its bounds without any complaints, as long as it is not
  ## dereferenced while outside its bounds.
  ##
  ## Due to the lack of constraints upon how the PyArrayRandAccIter's
  ## position can be changed, a PyArrayRandAccIter is slightly less
  ## predictable than a PyArrayForwardIter.  So prefer to use
  ## PyArrayForwardIter if you can.
  ##
  ## NOTE:  This proc requires that the PyArrayObject data is C-contiguous;
  ## else, an AssertionError will be raised.
  ##
  ## Here's an example of how you use this type of iterator:
  ##
  ##   let dt = arr.dtype
  ##   if dt == np_int32:
  ##       let bounds = arr.getBounds(int32)
  ##       var iter = arr.accessFlat(int32)
  ##       while iter in bounds:
  ##           doSomethingWith(iter[])
  ##           inc(iter, 2)
  ##           inc(iter, -1)
  ##

  # http://nim-lang.org/system.html#instantiationInfo,
  let ii = instantiationInfo()
  accessFlatImpl(arr, NimT, ii, "accessFlat", 0, 1)


template accessFlat*(arr: ptr PyArrayObject; NimT: typedesc[NumpyCompatibleNimType];
    incDelta: int): PyArrayRandAccIter[NimT] =
  # http://nim-lang.org/system.html#instantiationInfo,
  let ii = instantiationInfo()
  accessFlatImpl(arr, NimT, ii, "accessFlat", 0, incDelta)


template accessFlat*(arr: ptr PyArrayObject; NimT: typedesc[NumpyCompatibleNimType];
    initOffset, incDelta: int):
    PyArrayRandAccIter[NimT] =
  # http://nim-lang.org/system.html#instantiationInfo,
  let ii = instantiationInfo()
  accessFlatImpl(arr, NimT, ii, "accessFlat", initOffset, incDelta)


iterator items*[T](iter: PyArrayRandAccIter[T]):
    PyArrayRandAccIter[T] {.inline.} =
  let bounds = iter.getBounds()
  var iter = iter
  while iter in bounds:
    yield iter
    inc(iter)


proc getBoundsImpl(arr: ptr PyArrayObject, NimT: typedesc[NumpyCompatibleNimType],
    ii: InstantiationInfoTuple, procname: string{lit}):
    PyArrayIterBounds[NimT] =
  assertArrayType(arr, NimT, ii, procname)
  result = initPyArrayIterBounds[NimT](arr)


template getBounds*(arr: ptr PyArrayObject, NimT: typedesc[NumpyCompatibleNimType]):
    PyArrayIterBounds[NimT] =
  ## Return a PyArrayIterBounds over type `NimT`.

  # http://nim-lang.org/system.html#instantiationInfo,
  let ii = instantiationInfo()
  getBoundsImpl(arr, NimT, ii, "getBounds")


proc getBounds*[T](iter: PyArrayForwardIter[T]):
    PyArrayIterBounds[T] {.inline.} =
  ## Return a PyArrayIterBounds over type `T`.
  result = initPyArrayIterBounds(iter)


proc getBounds*[T](iter: PyArrayRandAccIter[T]):
    PyArrayIterBounds[T] {.inline.} =
  ## Return a PyArrayIterBounds over type `T`.
  result = initPyArrayIterBounds(iter)


## Data type descriptors:
##  http://docs.scipy.org/doc/numpy/reference/c-api.array.html#data-type-descriptors

proc getDescrFromType*(typenum: cint): ptr PyArrayDescr {.
    importc: "getDescrFromType", header: "pymodpkg/private/pyarrayobject_c.h", cdecl .}
  ## Returns a data-type object corresponding to `typenum`. The `typenum` can be
  ## one of the enumerated types, a character code for one of the enumerated types,
  ## or a user-defined type.
  ##
  ## http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_DescrFromType


template getDescrFromType*(typenum: CNpyTypes): ptr PyArrayDescr =
  getDescrFromType(ord(typenum))


template getDescrFromType*(nptype: NpType): ptr PyArrayDescr =
  getDescrFromType(ord(nptype.toCNpyTypes))


## Converting data types:
##  http://docs.scipy.org/doc/numpy/reference/c-api.array.html#converting-data-types

proc canCastArrayToImpl(arr: ptr PyArrayObject, totype: ptr PyArray_Descr, casting: cint):
    cint
    {. importc: "canCastArrayToImpl", header: "pymodpkg/private/pyarrayobject_c.h", cdecl .}
  ## Returns non-zero if `arr` can be cast to `totype` according to the casting
  ## rule given in `casting`.  If `arr` is an array scalar, its value is taken
  ## into account, and non-zero is also returned when the value will not overflow
  ## or be truncated to an integer when converting to a smaller type.
  ##
  ## http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_CanCastArrayTo


proc canCastArrayTo*(arr: ptr PyArrayObject, totype: ptr PyArray_Descr,
    casting: NpyCastingAlternatives): bool =
  return (canCastArrayToImpl(arr, totype, ord(casting)) != 0)


## Creating arrays:
##  http://docs.scipy.org/doc/numpy/reference/c-api.array.html#creating-arrays

proc createNewLikeArrayImpl(prototype: ptr PyArrayObject, order: cint,
    dtype: ptr PyArrayDescr, subok: cint): ptr PyArrayObject {.
    importc: "createNewLikeArrayImpl", header: "pymodpkg/private/pyarrayobject_c.h", cdecl .}
  ## This function steals a reference to `dtype` if it is not NULL.
  ##
  ## This array creation routine allows for the convenient creation of a new array
  ## matching an existing array’s shapes and memory layout, possibly changing the
  ## layout and/or data type.
  ##
  ## When `order` is NPY_ANYORDER, the result order is NPY_FORTRANORDER if
  ## `prototype` is a fortran array, NPY_CORDER otherwise.  When `order` is
  ## NPY_KEEPORDER, the result order matches that of `prototype`, even when
  ## the axes of `prototype` aren’t in C or Fortran order.
  ##
  ## If `dtype` is NULL, the data type of `prototype` is used.
  ##
  ## If `subok` is 1, the newly created array will use the sub-type of `prototype`
  ## to create the new array, otherwise it will create a base-class array.
  ##
  ## http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_NewLikeArray


template createNewLikeArray*(prototype: ptr PyArrayObject, order: NpyOrderAlternatives,
    dtype: ptr PyArrayDescr, subok: bool): ptr PyArrayObject =
  # http://nim-lang.org/system.html#instantiationInfo,
  let ii = instantiationInfo()
  registerNewPyObject(
      createNewLikeArrayImpl(prototype, ord(order), dtype, cint(subok)),
      WhereItCameFrom.AllocInNim, "createNewLikeArray", ii)


proc createSimpleNewImpl(nd: cint, dims: ptr npy_intp, typenum: cint): ptr PyArrayObject {.
    importc: "createSimpleNewImpl", header: "pymodpkg/private/pyarrayobject_c.h", cdecl .}
  ## Create a new unitialized array of type, `typenum`, whose size in each of
  ## `nd` dimensions is given by the integer array, `dims`.
  ##
  ## This function cannot be used to create a flexible-type array (no itemsize
  ## given).
  ##
  ## http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_SimpleNew


template createSimpleNew*(dims: CArrayProxy[npy_intp], nptype: NpType):
    ptr PyArrayObject =
  # http://nim-lang.org/system.html#instantiationInfo,
  let ii = instantiationInfo()
  registerNewPyObject(
      createSimpleNewImpl(dims.getLen, dims.getPtr, ord(nptype.toCNpyTypes)),
      WhereItCameFrom.AllocInNim, "createSimpleNew", ii)


proc assertNewShapeLengthLessEqualMaxDims(dims: openarray[int],
    created_at: InstantiationInfoTuple, procname: string) =
  # To see for yourself that Python-Numpy enforces a limit on the length of
  # the tuple of dimensions a caller is allowed to supply, run this little
  # Python script:
  #
  #     import numpy as np
  #     a = np.arange(1)
  #     t = ()
  #     for i in range(40):
  #         t = t + (1,)
  #         print "Shape:", t, len(t)
  #         b = a.reshape(t)
  #         print b, len(b.shape)
  #
  # It will raise a ValueError("sequence too large; must be smaller than 32")
  # when the length of `t` is 33 or larger.
  if dims.len > NPY_MAXDIMS:
    let msg = "$1: Supplied Numpy shape array is too long: supplied length == $2, but Numpy upper limit == $3 [File \"$4\", line $5]" %
        [procname, $(dims.len), $NPY_MAXDIMS, created_at.filename, $created_at.line]
    # http://nim-lang.org/docs/system.html#ValueError
    raise newException(ValueError, msg)

{.push warning[Uninit]: off.}

proc createSimpleNewOpenArrayImpl(dims: openarray[int], nptype: NpType,
    created_at: InstantiationInfoTuple, procname: string{lit}):
    ptr PyArrayObject =
  # Since we're passing an array into C, the size of (ie, number of bytes of)
  # each element will be used to step from one element to the next.
  # Hence, while `int` is a convenient element type for us to use in Nim
  # (since `int` is the type of an unadorned integer literal), we must ensure
  # that the array we pass into C is of the appropriately-sized element type.
  #
  # Create a new temporary array of `npy_intp` elements (the element type that
  # the Numpy C-API expects), in case `npy_intp` is not the same size as `int`.
  var dims_holder: array[NPY_MAXDIMS, npy_intp]

  # Complain if the shape specified by the caller has more dimensions than
  # `NPY_MAXDIMS` (the maximum number of dimensions that Numpy allows).
  assertNewShapeLengthLessEqualMaxDims(dims, created_at, procname)

  # We've already ensured that (dims.len <= NPY_MAXDIMS), so there's no chance
  # we can overrun the buffer here.
  let num_dims = min(dims.len, NPY_MAXDIMS)
  for i in 0.. <num_dims:
    dims_holder[i] = npy_intp(dims[i])

  let dims_ptr = addr(dims_holder[0])
  result = createSimpleNewImpl(cint(num_dims), dims_ptr, ord(nptype.toCNpyTypes))

{.pop.}  # {.push warning[Uninit]: off.}

template createSimpleNew*(dims: openarray[int], nptype: NpType):
    ptr PyArrayObject =
  # http://nim-lang.org/system.html#instantiationInfo,
  let ii = instantiationInfo()
  registerNewPyObject(
      createSimpleNewOpenArrayImpl(dims, nptype, ii, "createSimpleNew"),
      WhereItCameFrom.AllocInNim, "createSimpleNew", ii)


proc createNewCopyNewDataImpl(old: ptr PyArrayObject, order: cint): ptr PyArrayObject
    {. importc: "createNewCopyNewDataImpl", header: "pymodpkg/private/pyarrayobject_c.h", cdecl .}
  ## Equivalent to `ndarray.copy(self, fortran)`.  Make a copy of the `old` array.
  ## The returned array is always aligned and writeable with data interpreted the
  ## same as the `old` array.  If `order` is NPY_CORDER, then a C-style contiguous
  ## array is returned.  If `order` is NPY_FORTRANORDER, then a Fortran-style
  ## contiguous array is returned.  If `order` is NPY_ANYORDER, then the array
  ## returned is Fortran-style contiguous only if the old one is; otherwise, it is
  ## C-style contiguous.
  ##
  ## http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_NewCopy


template createNewCopyNewData*(old: ptr PyArrayObject, order: NpyOrderAlternatives):
    ptr PyArrayObject =
  # http://nim-lang.org/system.html#instantiationInfo,
  let ii = instantiationInfo()
  registerNewPyObject(
      createNewCopyNewDataImpl(old, ord(order)),
      WhereItCameFrom.AllocInNim, "createNewCopyNewData", ii)


## A convenient Python-like alias.
template copy*(old: ptr PyArrayObject): ptr PyArrayObject =
  let order: NpyOrderAlternatives = NpyOrderAlternatives.corder

  # http://nim-lang.org/system.html#instantiationInfo,
  let ii = instantiationInfo()
  registerNewPyObject(
      createNewCopyNewDataImpl(old, ord(order)),
      WhereItCameFrom.AllocInNim, "copy", ii)


proc doCopyIntoImpl(dest: ptr PyArrayObject, src: ptr PyArrayObject): cint {.
    importc: "doCopyIntoImpl", header: "pymodpkg/private/pyarrayobject_c.h", cdecl .}
  ## Copy from the source array, `src`, into the destination array, `dest`,
  ## performing a data-type conversion if necessary.  If an error occurs return -1
  ## (otherwise 0).  The shape of `src` must be broadcastable to the shape of `dest`.
  ## The data areas of `dest` and `src` must not overlap.
  ##
  ## http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_CopyInto


proc doCopyIntoRaiseOnError(dest: ptr PyArrayObject, src: ptr PyArrayObject,
    ii: InstantiationInfoTuple, procname: string{lit}) =
  let res: cint = doCopyIntoImpl(dest, src)
  if res != 0:
    let msg = "$1: Error during PyArray_CopyInto from type num $2 into type num $3 [File \"$4\", line $5]" %
        [procname, $(src.getDESCR().type_num), $(dest.getDESCR().type_num),
            ii.filename, $ii.line]
    # http://nim-lang.org/docs/system.html#ObjectConversionError
    raise newException(ObjectConversionError, msg)


template doCopyInto*(dest: ptr PyArrayObject, src: ptr PyArrayObject) =
  # http://nim-lang.org/system.html#instantiationInfo,
  let ii2 = instantiationInfo()
  doCopyIntoRaiseOnError(dest, src, ii2, "doCopyInto")


proc createAsTypeNewDataImpl(old: ptr PyArrayObject, newtype: NpType,
    ii: InstantiationInfoTuple, procname: string{lit}): ptr PyArrayObject =
  # This function is based loosely upon the ACTUAL, ORIGINAL `array_astype`
  # function in Numpy (exposed as `.astype(dtype)` in Python).
  #
  # (Source file: "numpy/core/src/multiarray/methods.c")
  let descr: ptr PyArrayDescr = getDescrFromType(newtype)
  if not canCastArrayTo(old, descr, NpyCastingAlternatives.unsafe_casting):
    let msg = "$1: Cannot cast array from type num $2 to type num $3 [File \"$4\", line $5]" %
        [procname, $(old.getDESCR().type_num), $(descr.type_num),
            ii.filename, $ii.line]
    # http://nim-lang.org/docs/system.html#ObjectConversionError
    raise newException(ObjectConversionError, msg)

  doPyIncRef(descr)
  result = registerNewPyObject(
      createNewLikeArrayImpl(old, ord(NpyOrderAlternatives.anyorder), descr, 1),
      WhereItCameFrom.AllocInNim, procname, ii)
  doCopyInto(result, old)


template createAsTypeNewData*(old: ptr PyArrayObject, newtype: NpType): ptr PyArrayObject =
  ## NOTE: This function will *always* create a copy of `old`, EVEN IF `newtype`
  ## is the same as the dtype of `old`.

  # http://nim-lang.org/system.html#instantiationInfo,
  let ii = instantiationInfo()
  createAsTypeNewDataImpl(old, newtype, ii, "createAsTypeNewData")


proc doFILLWBYTE*(arr: ptr PyArrayObject, val: cint) {.
    importc: "doFILLWBYTE", header: "pymodpkg/private/pyarrayobject_c.h", cdecl .}
  ## Fill the array pointed to by `arr` -- which must be a (subclass of)
  ## bigndarray -- with the contents of `val` (evaluated as a byte).
  ##
  ## This macro calls memset, so `arr` must be contiguous.
  ##
  ## http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_FILLWBYTE


proc doResizeDataInplaceImpl(old: ptr PyArrayObject, nd: cint, dims: ptr npy_intp, refcheck: cint) {.
    importc: "doResizeDataInplaceImpl", header: "pymodpkg/private/pyarrayobject_c.h", cdecl .}
  ## Equivalent to ndarray.resize(self, newshape, refcheck=refcheck).
  ##
  ## This function only works on single-segment arrays.  It changes the shape
  ## of `old` in-place and will reallocate the memory for `old` if `newshape`
  ## has a different total number of elements then the old shape.
  ##
  ## If reallocation is necessary, then `old` must own its data, and (unless
  ## `refcheck` is false) not be referenced by any other array.
  ##
  ## http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_Resize


{.push warning[Uninit]: off.}

proc doResizeDataInplaceOpenArrayImpl(old: ptr PyArrayObject, newShape: openarray[int], doRefCheck: bool,
    created_at: InstantiationInfoTuple, procname: string) =
  # Since we're passing an array into C, the size (ie, number of bytes)
  # of each element will be used to step from one element to the next.
  # Hence, while `int` is a convenient element type for us to use in Nim
  # (since `int` is the type of an unadorned integer literal), we must ensure
  # that the array we pass into C is of the appropriately-sized element.
  #
  # Create a new temporary array of `npy_intp` elements (the type that the
  # Numpy C-API expects), in case `npy_intp` is not the same size as `int`.
  var dims_holder: array[NPY_MAXDIMS, npy_intp]

  # Complain if the shape specified by the caller has more dimensions than
  # `NPY_MAXDIMS` (the maximum number of dimensions that Numpy allows).
  assertNewShapeLengthLessEqualMaxDims(newShape, created_at, procname)

  # We've already ensured that (dims.len <= NPY_MAXDIMS), so there's no chance
  # we can overrun the buffer here.
  let num_dims = min(newShape.len, NPY_MAXDIMS)
  for i in 0.. <num_dims:
    dims_holder[i] = npy_intp(newShape[i])

  let dims_ptr = addr(dims_holder[0])
  doResizeDataInplaceImpl(old, cint(num_dims), dims_ptr, cint(doRefCheck))

{.pop.}  # {.push warning[Uninit]: off.}


template doResizeDataInplace*(old: ptr PyArrayObject, newShape: openarray[int], doRefCheck: bool=true) =
  ## Equivalent to ndarray.resize(self, newshape, refcheck=refcheck).
  ##
  ## This function only works on single-segment arrays.  It changes the shape
  ## of `old` in-place and will reallocate the memory for `old` if `newshape`
  ## has a different total number of elements then the old shape.
  ##
  ## If reallocation is necessary, then `old` must own its data, and (unless
  ## `doRefCheck` is false) not be referenced by any other array.
  ##
  ## http://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.PyArray_Resize
  ##
  ## The docs for the Python equivalent `ndarray.resize` provide more info:
  ##  * "Only contiguous arrays can be resized."
  ##  * When decreasing the total size of an array, "array is flattened (in
  ##    the order that the data are stored in memory), resized, and reshaped."
  ##  * When enlarging an array, "missing entries are filled with zeros".
  ##
  ## http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.resize.html
  ##
  ## NOTE:  Despite what the C-API docs say, the C function `PyArray_Resize`
  ## actually returns `None`, _NOT_ "a reference to the new array"!
  ## (This was observed at line 158 of "numpy/core/src/multiarray/shape.c".)
  ## The Python docs correctly state that `None` is returned.

  # http://nim-lang.org/system.html#instantiationInfo,
  let ii = instantiationInfo()
  doResizeDataInplaceOpenArrayImpl(old, newShape, doRefCheck, ii, "doResizeDataInplace")


{.push warning[Uninit]: off.}

proc doResizeDataInplaceNumRows*(old: ptr PyArrayObject, newNumRows: int, doRefCheck: bool=true) =
  # Create a new temporary array of `npy_intp` elements (the type that
  # the Numpy C-API expects) containing the desired new shape.
  var dims_holder: array[NPY_MAXDIMS, npy_intp]

  # The number of dimensions must already be valid (or else the PyArrayObject
  # wouldn't exist!), so there's no chance we can overrun the buffer here.
  for i, d in old.enumerateDimensions:
    dims_holder[i] = d

  # Now update the temporary array with our new desired number of rows.
  dims_holder[0] = npy_intp(newNumRows)

  let dims_ptr = addr(dims_holder[0])
  doResizeDataInplaceImpl(old, old.ndim, dims_ptr, cint(doRefCheck))

{.pop.}  # {.push warning[Uninit]: off.}
