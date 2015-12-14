import strutils  # `%`
import pymod
import pymodpkg/docstrings
import pymodpkg/pyarrayobject


# for loop, values

proc int32FindMaxForLoopValues*(arr: ptr PyArrayObject): int32 {.exportpy} =
  result = low(int32)
  let dt = arr.dtype
  if dt == np_int32:
    for val in arr.values(int32):
      if val > result:
        result = val
  else:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)


# while loop, Forward Iter

proc int32FindMaxWhileLoopForwardIter*(arr: ptr PyArrayObject): int32 {.exportpy} =
  result = low(int32)
  let dt = arr.dtype
  if dt == np_int32:
    let bounds = arr.getBounds(int32)
    var iter = arr.iterateForward(int32)
    while iter in bounds:
      if iter[] > result:
        result = iter[]
      inc(iter)  # Increment the iterator manually.
  else:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)


# for loop, Forward Iter

proc int32FindMaxForLoopForwardIter*(arr: ptr PyArrayObject): int32 {.exportpy} =
  result = low(int32)
  let dt = arr.dtype
  if dt == np_int32:
    for iter in arr.iterateForward(int32):
      if iter[] > result:
        result = iter[]
  else:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)


# while loop, Rand Acc Iter

proc int32FindMaxWhileLoopRandaccIterDeref*(arr: ptr PyArrayObject): int32 {.exportpy} =
  result = low(int32)
  let dt = arr.dtype
  if dt == np_int32:
    let bounds = arr.getBounds(int32)
    var iter = arr.accessFlat(int32)
    while iter in bounds:
      if iter[] > result:
        result = iter[]
      inc(iter)  # Increment the iterator manually.
  else:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)


proc int32FindMaxWhileLoopRandaccIterIndex0*(arr: ptr PyArrayObject): int32 {.exportpy} =
  result = low(int32)
  let dt = arr.dtype
  if dt == np_int32:
    let bounds = arr.getBounds(int32)
    var iter = arr.accessFlat(int32)
    while iter in bounds:
      if iter[0] > result:
        result = iter[0]
      inc(iter)  # Increment the iterator manually.
  else:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)


proc int32FindMaxWhileLoopRandaccIterDeltaN_1*(arr: ptr PyArrayObject; n: int): int32 {.exportpy} =
  result = low(int32)
  let dt = arr.dtype
  if dt == np_int32:
    let bounds = arr.getBounds(int32)
    var iter = arr.accessFlat(int32)
    while iter in bounds:
      if iter[] > result:
        result = iter[]
      inc(iter, n)  # Increment the iterator manually.
  else:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)


#proc int32FindMaxWhileLoopRandaccIterDeltaN_2*(arr: ptr PyArrayObject; n: int): int32 {.exportpy} =
#  result = low(int32)
#  let dt = arr.dtype
#  if dt == np_int32:
#    let bounds = arr.getBounds(int32)
#    var iter = arr.accessFlat(int32, n)
#    while iter in bounds:
#      if iter[] > result:
#        result = iter[]
#      inc(iter)  # Increment the iterator manually.
#  else:
#    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
#    raise newException(ValueError, msg)


proc int32FindMaxWhileLoopRandaccIterExcludeFirstM_1*(arr: ptr PyArrayObject; m: int): int32 {.exportpy} =
  result = low(int32)
  let dt = arr.dtype
  if dt == np_int32:
    let bounds = arr.getBounds(int32)
    var iter = arr.accessFlat(int32) + m
    while iter in bounds:
      if iter[0] > result:
        result = iter[0]
      inc(iter)  # Increment the iterator manually.
  else:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)


#proc int32FindMaxWhileLoopRandaccIterExcludeFirstM_2*(arr: ptr PyArrayObject; m: int): int32 {.exportpy} =
#  result = low(int32)
#  let dt = arr.dtype
#  if dt == np_int32:
#    let bounds = arr.getBounds(int32)
#    var iter = arr.accessFlat(int32, 1, m)
#    while iter in bounds:
#      if iter[0] > result:
#        result = iter[0]
#      inc(iter)  # Increment the iterator manually.
#  else:
#    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
#    raise newException(ValueError, msg)


proc int32FindMaxWhileLoopRandaccIterExcludeLastM_1*(arr: ptr PyArrayObject; m: int): int32 {.exportpy} =
  result = low(int32)
  let dt = arr.dtype
  if dt == np_int32:
    let bounds = arr.getBounds(int32)
    var iter = arr.accessFlat(int32) + m
    while iter in bounds:
      if iter[-m] > result:
        result = iter[-m]
      inc(iter)  # Increment the iterator manually.
  else:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)


#proc int32FindMaxWhileLoopRandaccIterExcludeLastM_2*(arr: ptr PyArrayObject; m: int): int32 {.exportpy} =
#  result = low(int32)
#  let dt = arr.dtype
#  if dt == np_int32:
#    let bounds = arr.getBounds(int32)
#    var iter = arr.accessFlat(int32, 1, m)
#    while iter in bounds:
#      if iter[-m] > result:
#        result = iter[-m]
#      inc(iter)  # Increment the iterator manually.
#  else:
#    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
#    raise newException(ValueError, msg)


# for loop, Rand Acc Iter

proc int32FindMaxForLoopRandAccIterDeref*(arr: ptr PyArrayObject): int32 {.exportpy} =
  result = low(int32)
  let dt = arr.dtype
  if dt == np_int32:
    for iter in arr.accessFlat(int32):
      if iter[] > result:
        result = iter[]
  else:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)


proc int32FindMaxForLoopRandAccIterIndex0*(arr: ptr PyArrayObject): int32 {.exportpy} =
  result = low(int32)
  let dt = arr.dtype
  if dt == np_int32:
    for iter in arr.accessFlat(int32):
      if iter[0] > result:
        result = iter[0]
  else:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)


proc int32FindMaxForLoopRandaccIterDeltaN*(arr: ptr PyArrayObject; n: int): int32 {.exportpy} =
  result = low(int32)
  let dt = arr.dtype
  if dt == np_int32:
    for iter in arr.accessFlat(int32, n):
      if iter[] > result:
        result = iter[]
  else:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)


proc int32FindMaxForLoopRandaccIterExcludeFirstM*(arr: ptr PyArrayObject; m: int): int32 {.exportpy} =
  result = low(int32)
  let dt = arr.dtype
  if dt == np_int32:
    for iter in arr.accessFlat(int32, 1, m):
      if iter[0] > result:
        result = iter[0]
  else:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)


proc int32FindMaxForLoopRandaccIterExcludeLastM*(arr: ptr PyArrayObject; m: int): int32 {.exportpy} =
  result = low(int32)
  let dt = arr.dtype
  if dt == np_int32:
    for iter in arr.accessFlat(int32, 1, m):
      if iter[-m] > result:
        result = iter[-m]
  else:
    let msg = "expected input array of dtype $1, received dtype $2" % [$np_int32, $dt]
    raise newException(ValueError, msg)


initPyModule("", 
    # for loop, values
    int32FindMaxForLoopValues,
    # while loop, Forward Iter
    int32FindMaxWhileLoopForwardIter,
    # for loop, Forward Iter
    int32FindMaxForLoopForwardIter,
    # while loop, Rand Acc Iter
    int32FindMaxWhileLoopRandaccIterDeref, int32FindMaxWhileLoopRandaccIterIndex0,
    int32FindMaxWhileLoopRandaccIterDeltaN_1, #int32FindMaxWhileLoopRandaccIterDeltaN_2,
    int32FindMaxWhileLoopRandaccIterExcludeFirstM_1, #int32FindMaxWhileLoopRandaccIterExcludeFirstM_2,
    int32FindMaxWhileLoopRandaccIterExcludeLastM_1, #int32FindMaxWhileLoopRandaccIterExcludeLastM_2,
    # for loop, Rand Acc Iter
    int32FindMaxForLoopRandAccIterDeref, int32FindMaxForLoopRandAccIterIndex0,
    int32FindMaxForLoopRandaccIterDeltaN,
    int32FindMaxForLoopRandaccIterExcludeFirstM,
    int32FindMaxForLoopRandaccIterExcludeLastM)
