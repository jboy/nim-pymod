import strutils  # `%`
import pymod
import pymodpkg/miscutils
import pymodpkg/pyarrayobject


proc returnPyArrayObjectPtrAsInt*(arr: ptr PyArrayObject): int {.exportpy.} = cast[int](arr)

proc returnDtypeAsString*(arr: ptr PyArrayObject): string {.exportpy.} = $arr.dtype

proc returnDataPointerAsInt*(arr: ptr PyArrayObject): int {.exportpy.} = cast[int](arr.data)


proc returnBoolDataPtrAsInt*(arr: ptr PyArrayObject): int {.exportpy.} =
  let dt = arr.dtype
  if dt != np_bool:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int8, $dt]
    raise newException(ValueError, msg)
  else:
    result = cast[int](arr.data(bool))

proc returnInt8DataPtrAsInt*(arr: ptr PyArrayObject): int {.exportpy.} =
  let dt = arr.dtype
  if dt != np_int8:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int8, $dt]
    raise newException(ValueError, msg)
  else:
    result = cast[int](arr.data(int8))

proc returnInt16DataPtrAsInt*(arr: ptr PyArrayObject): int {.exportpy.} =
  let dt = arr.dtype
  if dt != np_int16:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int16, $dt]
    raise newException(ValueError, msg)
  else:
    result = cast[int](arr.data(int16))

proc returnInt32DataPtrAsInt*(arr: ptr PyArrayObject): int {.exportpy.} =
  let dt = arr.dtype
  if dt != np_int32:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)
  else:
    result = cast[int](arr.data(int32))

proc returnInt64DataPtrAsInt*(arr: ptr PyArrayObject): int {.exportpy.} =
  let dt = arr.dtype
  if dt != np_int64:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int64, $dt]
    raise newException(ValueError, msg)
  else:
    result = cast[int](arr.data(int64))

proc returnFloat32DataPtrAsInt*(arr: ptr PyArrayObject): int {.exportpy.} =
  let dt = arr.dtype
  if dt != np_float32:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)
  else:
    result = cast[int](arr.data(float32))

proc returnFloat64DataPtrAsInt*(arr: ptr PyArrayObject): int {.exportpy.} =
  let dt = arr.dtype
  if dt != np_float64:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int64, $dt]
    raise newException(ValueError, msg)
  else:
    result = cast[int](arr.data(float64))


#proc returnBoolDataPtrIndex0*(arr: ptr PyArrayObject): bool {.exportpy.} =
#  let dt = arr.dtype
#  if dt != np_bool:
#    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int8, $dt]
#    raise newException(ValueError, msg)
#  else:
#    result = arr.data(bool)[]

#proc returnInt8DataPtrIndex0*(arr: ptr PyArrayObject): int8 {.exportpy.} =
#  let dt = arr.dtype
#  if dt != np_int8:
#    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int8, $dt]
#    raise newException(ValueError, msg)
#  else:
#    result = arr.data(int8)[]

proc returnInt16DataPtrIndex0*(arr: ptr PyArrayObject): int16 {.exportpy.} =
  let dt = arr.dtype
  if dt != np_int16:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int16, $dt]
    raise newException(ValueError, msg)
  else:
    result = arr.data(int16)[]

proc returnInt32DataPtrIndex0*(arr: ptr PyArrayObject): int32 {.exportpy.} =
  let dt = arr.dtype
  if dt != np_int32:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)
  else:
    result = arr.data(int32)[]

proc returnInt64DataPtrIndex0*(arr: ptr PyArrayObject): int64 {.exportpy.} =
  let dt = arr.dtype
  if dt != np_int64:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int64, $dt]
    raise newException(ValueError, msg)
  else:
    result = arr.data(int64)[]

proc returnFloat32DataPtrIndex0*(arr: ptr PyArrayObject): float32 {.exportpy.} =
  let dt = arr.dtype
  if dt != np_float32:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)
  else:
    result = arr.data(float32)[]

proc returnFloat64DataPtrIndex0*(arr: ptr PyArrayObject): float64 {.exportpy.} =
  let dt = arr.dtype
  if dt != np_float64:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int64, $dt]
    raise newException(ValueError, msg)
  else:
    result = arr.data(float64)[]


proc returnNdAttr*(arr: ptr PyArrayObject): cint {.exportpy.} = arr.nd

proc returnNdimAttr*(arr: ptr PyArrayObject): cint {.exportpy.} = arr.ndim


# FIXME:  Technically, this really should return `npy_intp` rather than `int`.
# Yes, the two types are identical (a signed integer that is the same size as
# a pointer), but still... technically...
#proc returnDimensionsAsTuple1D*(arr: ptr PyArrayObject): tuple[a: npy_intp] {.exportpy.} =
proc returnDimensionsAsTuple1D*(arr: ptr PyArrayObject): tuple[a: int] {.exportpy.} =
  let expected_nd = 1
  if arr.nd != expected_nd:
    let msg = "expected input array of ndim=$1, received ndim=$2" % [$expected_nd, $arr.nd]
    raise newException(ValueError, msg)
  else:
    result = (a: arr.dimensions[0])

# FIXME:  Technically, this really should return `npy_intp` rather than `int`.
# Yes, the two types are identical (a signed integer that is the same size as
# a pointer), but still... technically...
#proc returnShapeAsTuple1D*(arr: ptr PyArrayObject): tuple[a: npy_intp] {.exportpy.} =
proc returnShapeAsTuple1D*(arr: ptr PyArrayObject): tuple[a: int] {.exportpy.} =
  let expected_nd = 1
  if arr.nd != expected_nd:
    let msg = "expected input array of ndim=$1, received ndim=$2" % [$expected_nd, $arr.nd]
    raise newException(ValueError, msg)
  else:
    result = (a: arr.shape[0])


# FIXME:  Technically, this really should return `npy_intp` rather than `int`.
# Yes, the two types are identical (a signed integer that is the same size as
# a pointer), but still... technically...
#proc returnDimensionsAsTuple2D*(arr: ptr PyArrayObject): tuple[a, b: npy_intp] {.exportpy.} =
proc returnDimensionsAsTuple2D*(arr: ptr PyArrayObject): tuple[a, b: int] {.exportpy.} =
  let expected_nd = 2
  if arr.nd != expected_nd:
    let msg = "expected input array of ndim=$1, received ndim=$2" % [$expected_nd, $arr.nd]
    raise newException(ValueError, msg)
  else:
    result = (arr.dimensions[0], arr.dimensions[1])

# FIXME:  Technically, this really should return `npy_intp` rather than `int`.
# Yes, the two types are identical (a signed integer that is the same size as
# a pointer), but still... technically...
#proc returnShapeAsTuple2D*(arr: ptr PyArrayObject): tuple[a, b: npy_intp] {.exportpy.} =
proc returnShapeAsTuple2D*(arr: ptr PyArrayObject): tuple[a, b: int] {.exportpy.} =
  let expected_nd = 2
  if arr.nd != expected_nd:
    let msg = "expected input array of ndim=$1, received ndim=$2" % [$expected_nd, $arr.nd]
    raise newException(ValueError, msg)
  else:
    result = (arr.shape[0], arr.shape[1])


# FIXME:  Technically, this really should return `npy_intp` rather than `int`.
# Yes, the two types are identical (a signed integer that is the same size as
# a pointer), but still... technically...
#proc returnDimensionsAsTuple3D*(arr: ptr PyArrayObject): tuple[a, b, c: npy_intp] {.exportpy.} =
proc returnDimensionsAsTuple3D*(arr: ptr PyArrayObject): tuple[a, b, c: int] {.exportpy.} =
  let expected_nd = 3
  if arr.nd != expected_nd:
    let msg = "expected input array of ndim=$1, received ndim=$2" % [$expected_nd, $arr.nd]
    raise newException(ValueError, msg)
  else:
    result = (arr.dimensions[0], arr.dimensions[1], arr.dimensions[2])

# FIXME:  Technically, this really should return `npy_intp` rather than `int`.
# Yes, the two types are identical (a signed integer that is the same size as
# a pointer), but still... technically...
#proc returnShapeAsTuple3D*(arr: ptr PyArrayObject): tuple[a, b, c: npy_intp] {.exportpy.} =
proc returnShapeAsTuple3D*(arr: ptr PyArrayObject): tuple[a, b, c: int] {.exportpy.} =
  let expected_nd = 3
  if arr.nd != expected_nd:
    let msg = "expected input array of ndim=$1, received ndim=$2" % [$expected_nd, $arr.nd]
    raise newException(ValueError, msg)
  else:
    result = (arr.shape[0], arr.shape[1], arr.shape[2])


# FIXME:  Technically, this really should return `npy_intp` rather than `int`.
# Yes, the two types are identical (a signed integer that is the same size as
# a pointer), but still... technically...
#proc returnDimensionsAsTuple4D*(arr: ptr PyArrayObject): tuple[a, b, c, d: npy_intp] {.exportpy.} =
proc returnDimensionsAsTuple4D*(arr: ptr PyArrayObject): tuple[a, b, c, d: int] {.exportpy.} =
  let expected_nd = 4
  if arr.nd != expected_nd:
    let msg = "expected input array of ndim=$1, received ndim=$2" % [$expected_nd, $arr.nd]
    raise newException(ValueError, msg)
  else:
    result = (arr.dimensions[0], arr.dimensions[1], arr.dimensions[2], arr.dimensions[3])

# FIXME:  Technically, this really should return `npy_intp` rather than `int`.
# Yes, the two types are identical (a signed integer that is the same size as
# a pointer), but still... technically...
#proc returnShapeAsTuple4D*(arr: ptr PyArrayObject): tuple[a, b, c, d: npy_intp] {.exportpy.} =
proc returnShapeAsTuple4D*(arr: ptr PyArrayObject): tuple[a, b, c, d: int] {.exportpy.} =
  let expected_nd = 4
  if arr.nd != expected_nd:
    let msg = "expected input array of ndim=$1, received ndim=$2" % [$expected_nd, $arr.nd]
    raise newException(ValueError, msg)
  else:
    result = (arr.shape[0], arr.shape[1], arr.shape[2], arr.shape[3])


initPyModule("",
    returnPyArrayObjectPtrAsInt, returnDtypeAsString, returnDataPointerAsInt,
    returnBoolDataPtrAsInt, returnInt8DataPtrAsInt, returnInt16DataPtrAsInt,
    returnInt32DataPtrAsInt, returnInt64DataPtrAsInt,
    returnFloat32DataPtrAsInt, returnFloat64DataPtrAsInt,
    returnInt16DataPtrIndex0, returnInt32DataPtrIndex0, returnInt64DataPtrIndex0,
    returnFloat32DataPtrIndex0, returnFloat64DataPtrIndex0,
    returnNdAttr, returnNdimAttr,
    returnDimensionsAsTuple1D, returnShapeAsTuple1D,
    returnDimensionsAsTuple2D, returnShapeAsTuple2D,
    returnDimensionsAsTuple3D, returnShapeAsTuple3D,
    returnDimensionsAsTuple4D, returnShapeAsTuple4D)

