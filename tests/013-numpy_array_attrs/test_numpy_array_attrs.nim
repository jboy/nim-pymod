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


initPyModule("",
    returnPyArrayObjectPtrAsInt, returnDtypeAsString, returnDataPointerAsInt,
    returnBoolDataPtrAsInt, returnInt8DataPtrAsInt, returnInt16DataPtrAsInt,
    returnInt32DataPtrAsInt, returnInt64DataPtrAsInt,
    returnFloat32DataPtrAsInt, returnFloat64DataPtrAsInt)
