# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

## Membrain implements a very simple garbage collector for PyObjects in Pymod.
##
## This frees the Pymod user from the need to increment or decrement PyObject
## reference counts manually (and more specifically, from the need to decide
## **when** ref-counts should be incremented or decremented):  The Pymod user
## simply creates new PyObjects or PyArrayObjects as needed (using one of the
## Python or Numpy object-creation functions that has been exposed in Pymod);
## Membrain will keep track of the objects and decrement their ref-counts when
## appropriate to avoid memory leaks or dangling pointers.
##
## How does it work?  Membrain is the third-laziest garbage collector possible.
## (The first two being "wait for operating system shutdown" and "wait for
## program termination", respectively.)  Membrain does its garbage collection
## when the Pymod-wrapped Nim proc returns control to Python.
##
## Membrain is a thin, permeable layer (get it?) between Nim and Python/C-API,
## that separates the PyObject allocations in Nim from the PyObject allocations
## in Python:  When the Pymod-wrapped Nim proc returns control to Python, the
## only PyObjects that will survive will be (1) those that were passed into the
## proc from Python, and (2) those that are returned from the proc to Python.
## All others will have their ref-counts decremented -- which, since they were
## allocated in Nim with a newborn ref-count of 1, will leave them all with
## ref-counts of zero, causing them to be deallocated.

import strutils

import pymodpkg/miscutils
import pymodpkg/pyobject


const DoPrintDebugInfo = false


# Currently we only register & manage PyObjects that were allocated in Nim.
# If needed in the future, this enum will enable us to register & differentiate
# PyObjects obtained from other sources (eg, passed in as a C-API argument from
# Python, extracted from a Python collection, etc.)
type WhereItCameFrom* {. pure .} = enum
  AllocInPython = "AllocInPython",
  AllocInNim = "AllocInNim",

# http://nim-lang.org/system.html#instantiationInfo,
type InstantiationInfoTuple = tuple[filename: string, line: int]

type RegisteredPyObjectInfo = object
  from_where: WhereItCameFrom
  which_func: string  # name of the allocation func that created the PyObject
  created_at: InstantiationInfoTuple  # where the func was called

type RegisteredPyObject* = object
  # Instances `RegisteredPyObject` will be contained directly in the `seq`,
  # to minimise the amount of indirection needed to obtain the `ptr PyObject`
  # value for pointer-equality comparisons in for-loops.
  #
  # Since `RegisteredPyObject` only contains a `ptr` and a `ref`, it will be
  # very inexpensive to copy instances of `RegisteredPyObject` when the `seq`
  # needs to be resized.
  obj: ptr PyObject
  info: ref RegisteredPyObjectInfo


var RegisteredPyObjects: seq[RegisteredPyObject] = @[]


proc initRegisteredPyObjects*() =
  RegisteredPyObjects = @[]


when DoPrintDebugInfo:
  template echoInfo(rpo: RegisteredPyObject, context: string) =
    echo(context, " ", rpo.obj.toHex)
    let info = rpo.info
    echo(" - from_where = ", $info.from_where)
    echo(" - which_func = ", info.which_func)
    echo(" - $1:$2" % [info.created_at.filename, $info.created_at.line])
    echo(" - refcount = ", getPyRefCnt(rpo.obj))


proc registerNewPyObjectImpl*(obj: ptr PyObject,
    from_where: WhereItCameFrom, which_func: string,
    created_at: InstantiationInfoTuple): ptr PyObject =
  var info: ref RegisteredPyObjectInfo
  new(info)
  info.from_where = from_where
  info.which_func = which_func
  info.created_at = created_at

  # FIXME:  Can / should we handle memory-allocation failures here?
  # There are various "PyObject-allocating" procs:  createNewLikeArray,
  # createSimpleNew, createNewCopy, etc.  They all invoke this proc to
  # register their PyObject.  What do the Numpy C-API functions inside
  # these procs return when malloc fails?  Do they return NULL pointers?
  # TODO:  Look this up and handle malloc failures appropriately.

  var rpo: RegisteredPyObject = RegisteredPyObject(obj: obj, info: info)
  RegisteredPyObjects.add(rpo)
  when DoPrintDebugInfo:
    rpo.echoInfo("\nRegister PyObject")

  return obj


# This generic proc assumes that `T` is implicitly convertible to PyObject.
# An example such `T` is PyArrayObject.
#
# However, we need to cast *back* to `ptr T`, because in general, PyObject
# should NOT be implicitly convertible back to its derived types.
proc registerNewPyObject*[T](obj: ptr T,
    from_where: WhereItCameFrom, which_func: string,
    created_at: InstantiationInfoTuple): ptr T =
  let py_obj: ptr PyObject = obj  # <- implicit conversion occurs here
  return cast[ptr T](registerNewPyObjectImpl(py_obj, from_where, which_func, created_at))


proc decRefAllRegisteredPyObjects*() =
  when DoPrintDebugInfo:
    echo("\ndecRefAllRegisteredPyObjects()...")
  while RegisteredPyObjects.len > 0:
    let rpo = RegisteredPyObjects.pop()
    when DoPrintDebugInfo:
      rpo.echoInfo("Processing registered PyObject")

# We don't register PyObjects that were passed in from Python.
#    if rpo.from_where == WhereItCameFrom.AllocInPython:
#      # This PyObject was passed in from Python through the C API.
#      #
#      # Python manages the ref-counts of such PyObjects just fine already
#      # (ie, without any interference from us), so we shouldn't break
#      # that correctness.  Basically, the ref-count of this PyObject
#      # should be the same when we leave Nim as it was when we entered.
#      #
#      # Since we didn't increment its ref-count when the PyObject passed
#      # into Nim (and the "O" format code passed to `PyArg_ParseTuple`
#      # also doesn't cause the object's ref-count to be increased), we
#      # shouldn't decrement the ref-count when the PyObject leaves Nim.
#      # So, nothing for us to do here.
#      continue

    when DoPrintDebugInfo:
      echo(" > refcount -> ", getPyRefCnt(rpo.obj) - 1)
    doPyDecRef(rpo.obj)


proc findRegisteredPyObjectByValue*[T](obj: ptr T): ref RegisteredPyObject =
  let cast_obj = cast[ptr PyObject](obj)
  for rpo in RegisteredPyObjects:
    if rpo.obj == cast_obj:
      # Yes, we've found our object.
      return rpo

  # Otherwise, no match.
  return nil


proc collectAllGarbage*() =
  when DoPrintDebugInfo:
    echo("\ncollectAllGarbage()...")
  decRefAllRegisteredPyObjects()
  GC_fullCollect()  # http://nim-lang.org/system.html#GC_fullCollect

