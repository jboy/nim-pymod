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


initPyModule("",
    returnPyArrayObjectPtrAsInt, returnDtypeAsString, returnDataPointerAsInt,
    returnBoolDataPtrAsInt, returnInt8DataPtrAsInt, returnInt16DataPtrAsInt,
    returnInt32DataPtrAsInt, returnInt64DataPtrAsInt,
    returnFloat32DataPtrAsInt, returnFloat64DataPtrAsInt,
    returnInt16DataPtrIndex0, returnInt32DataPtrIndex0, returnInt64DataPtrIndex0,
    returnFloat32DataPtrIndex0, returnFloat64DataPtrIndex0)
