import strutils  # `%`
import pymod
import pymodpkg/miscutils
import pymodpkg/pyarrayobject


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
    returnDimensionsAsTuple1D, returnShapeAsTuple1D,
    returnDimensionsAsTuple2D, returnShapeAsTuple2D,
    returnDimensionsAsTuple3D, returnShapeAsTuple3D,
    returnDimensionsAsTuple4D, returnShapeAsTuple4D)

