import strutils  # `%`
import pymod
import pymodpkg/miscutils
import pymodpkg/pyarrayobject


# FIXME:  Technically, this really should return `npy_intp` rather than `int`.
# Yes, the two types are identical (a signed integer that is the same size as
# a pointer), but still... technically...
#proc returnStridesAsTuple1D*(arr: ptr PyArrayObject): tuple[a: npy_intp] {.exportpy.} =
proc returnStridesAsTuple1D*(arr: ptr PyArrayObject): tuple[a: int] {.exportpy.} =
  let expected_nd = 1
  if arr.nd != expected_nd:
    let msg = "expected input array of ndim=$1, received ndim=$2" % [$expected_nd, $arr.nd]
    raise newException(ValueError, msg)
  else:
    result = (a: arr.strides[0])


# FIXME:  Technically, this really should return `npy_intp` rather than `int`.
# Yes, the two types are identical (a signed integer that is the same size as
# a pointer), but still... technically...
#proc returnStridesAsTuple2D*(arr: ptr PyArrayObject): tuple[a, b: npy_intp] {.exportpy.} =
proc returnStridesAsTuple2D*(arr: ptr PyArrayObject): tuple[a, b: int] {.exportpy.} =
  let expected_nd = 2
  if arr.nd != expected_nd:
    let msg = "expected input array of ndim=$1, received ndim=$2" % [$expected_nd, $arr.nd]
    raise newException(ValueError, msg)
  else:
    result = (arr.strides[0], arr.strides[1])


# FIXME:  Technically, this really should return `npy_intp` rather than `int`.
# Yes, the two types are identical (a signed integer that is the same size as
# a pointer), but still... technically...
#proc returnStridesAsTuple3D*(arr: ptr PyArrayObject): tuple[a, b, c: npy_intp] {.exportpy.} =
proc returnStridesAsTuple3D*(arr: ptr PyArrayObject): tuple[a, b, c: int] {.exportpy.} =
  let expected_nd = 3
  if arr.nd != expected_nd:
    let msg = "expected input array of ndim=$1, received ndim=$2" % [$expected_nd, $arr.nd]
    raise newException(ValueError, msg)
  else:
    result = (arr.strides[0], arr.strides[1], arr.strides[2])


# FIXME:  Technically, this really should return `npy_intp` rather than `int`.
# Yes, the two types are identical (a signed integer that is the same size as
# a pointer), but still... technically...
#proc returnStridesAsTuple4D*(arr: ptr PyArrayObject): tuple[a, b, c, d: npy_intp] {.exportpy.} =
proc returnStridesAsTuple4D*(arr: ptr PyArrayObject): tuple[a, b, c, d: int] {.exportpy.} =
  let expected_nd = 4
  if arr.nd != expected_nd:
    let msg = "expected input array of ndim=$1, received ndim=$2" % [$expected_nd, $arr.nd]
    raise newException(ValueError, msg)
  else:
    result = (arr.strides[0], arr.strides[1], arr.strides[2], arr.strides[3])


initPyModule("",
    returnStridesAsTuple1D, returnStridesAsTuple2D,
    returnStridesAsTuple3D, returnStridesAsTuple4D)

