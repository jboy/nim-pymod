import strutils  # `%`
import pymod
import pymodpkg/pyarrayobject


proc returnDtypeAsString*(arr: ptr PyArrayObject): string {.exportpy.} = $arr.dtype

proc returnDataPointerAsInt*(arr: ptr PyArrayObject): int {.exportpy.} = cast[int](arr.data)

proc returnInt16DataPtrAsInt*(arr: ptr PyArrayObject): int {.exportpy.} =
  let dt = arr.dtype
  if dt != np_int16:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int16, $dt]
    raise newException(ValueError, msg)
  else:
    result = cast[int](arr.data(int16))


initPyModule("",
    returnDtypeAsString, returnDataPointerAsInt, returnInt16DataPtrAsInt)
