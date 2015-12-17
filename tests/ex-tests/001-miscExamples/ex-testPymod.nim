# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import pymod
import pymodpkg/docstrings
import pymodpkg/pyarrayobject


##
##=== An example of howto use Pymod.
##

proc myfunc1*(x: int, y: string, z: float64) {. exportpy .} =
  ## A double-hashed comment at the top of the proc's body is the correct way
  ## to do docstrings in Nim.  It's recognised by Nim's "docgen" utility, which
  ## is how the official Nim docs are generated.
  ##
  ## This double-hashed comment shows up in the AST as a `CommentStmt` at the
  ## top of the `StmtList` that is the proc's body.  The Nimrod node kind is
  ## `nnkCommentStmt`.
  ##
  ## HOWEVER, I can't for the life of me work out how to actually ACCESS the
  ## docstring content from a CommentStmt node in the AST.  It looks like the
  ## content is discarded at the PNode phase in "Nim/compiler/renderer.nim".
  ## [The Nim compiler parser appears to produce PNode instances, which are
  ## defined in "Nim/compiler/ast.nim" and have a string `comment` field.
  ## These PNode instances are then converted to TNimrodNode instances for
  ## the macros to process, but the TNimrodNode type is field-less and opaque
  ## in Nim.]  Boo!
  ##
  ## [Update at a later date:  I got confirmation from Araq that it's not
  ## currently possible to access the content of double-hashed comments in
  ## the Nim AST:  https://github.com/nim-lang/Nim/issues/2024 ]
  ##
  ## So, instead I'm going to mimic Python-style docstrings:  Insert a
  ## triple-quoted string literal at the top of the proc's body (after
  ## any double-hashed comments that are present for Nim's docgen) and
  ## apply the `pydocstring` pragma to the proc!
  docstring"""This is a Python-style docstring!

  It doesn't do anything by itself, but it can be extracted by pragmas
  (like the `exportpy` pragma, for example) that process this proc.
  """
  echo($x & y & $z)

  docstring"""This is another Python-style docstring in the same proc.
  There can be any number of docstrings amongst the top-level statements
  of a proc.
  """

proc otherfunc1*(a: int, b: ptr PyObject): ptr PyObject {. exportpy .} =
  echo($a)
  echo(repr(b))
  return b

proc simpleAdd1*(x, y, z: int): int {. exportpy .} =
  return (x + y + z)

when isMainModule:
  myfunc1(4, "hello", 5.5)
  discard otherfunc1(4, nil)
  discard simpleAdd1(4, 3, 5)


proc myfunc2*(x: int, y: string, z: float64) {. exportpy .} =
  echo($x & y & $z)

proc otherfunc2*(a: int, b: ptr PyObject): ptr PyObject {. exportpy .} =
  echo($a)
  echo(repr(b))
  return b

proc simpleAdd2*(x, y, z: int): int {. exportpy .} =
  return (x + y + z)

when isMainModule:
  myfunc2(4, "hello", 5.5)
  discard otherfunc2(4, nil)
  discard simpleAdd2(15, 17, 22)


proc myfunc3*(x: int, y: string, z: float64) {. exportpy .} =
  echo($x & y & $z)

proc otherfunc3*(a: int, b: ptr PyObject): ptr PyObject {. exportpy .} =
  echo($a)
  echo(repr(b))
  return b


proc myNumpyAdd*(arr: ptr PyArrayObject, x: int): ptr PyArrayObject {. exportpy .} =
  echo("ptr = ", cast[int](arr))

  # Test out the Array C API macro-style interface:
  #  http://docs.scipy.org/doc/numpy/reference/c-api.array.html#array-structure-and-data-access
  echo("PyArray_NDIM = ", getNDIM(arr))
  echo("PyArray_DIMS = ", repr(getDIMS(arr)))
  echo("PyArray_SHAPE = ", repr(getSHAPE(arr)))
  echo("PyArray_DATA = ", repr(getDATA(arr)))
  echo("PyArray_STRIDES = ", repr(getSTRIDES(arr)))
  echo("PyArray_FLAGS = ", getFLAGS(arr))
  echo("PyArray_ITEMSIZE = ", getITEMSIZE(arr))
  echo("PyArray_TYPE = ", getTYPE(arr))

  # Test out the PyArrayObject struct attribute-style interface:
  #  http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArrayObject
  echo("arr.nd = ", arr.nd)
  echo("dims =")
  for d in arr.dimensions:
    echo("  : ", d)
  echo("dims =")
  for t in arr.enumerateDimensions:
    echo("  : ", t)
  echo("strides =")
  for s in arr.strides:
    echo("  : ", s)
  echo("strides =")
  for t in arr.enumerateStrides:
    echo("  : ", t)

  if arr.nd > 0:
    echo("arr.dimensions[0] = ", arr.dimensions[0])
    echo("arr.strides[0] = ", arr.strides[0])
  if arr.nd > 1:
    echo("arr.dimensions[1] = ", arr.dimensions[1])
    echo("arr.strides[1] = ", arr.strides[1])
  if arr.nd > 2:
    echo("arr.dimensions[2] = ", arr.dimensions[2])
    echo("arr.strides[2] = ", arr.strides[2])

  echo("flags = ", arr.flags)

  let d = arr.descr
  echo("arr.descr.dkind = ", d.dkind)
  echo("arr.descr.dtype = ", d.dtype)
  echo("arr.descr.byteorder = ", d.byteorder)
  echo("arr.descr.dflags = ", d.dflags)
  echo("arr.descr.type_num = ", d.type_num)
  echo("arr.descr.elsize = ", d.elsize)
  echo("arr.descr.alignment = ", d.alignment)
  echo("arr.dtype = ", arr.dtype)

  let dt = arr.dtype
  if dt == np_int64:
    echo("\narr.dtype is ", $dt)
    echo("iterate through arr elems:")
    var iter = arr.iterateFlat(int64)
    let bounds = arr.getBounds(int64)
    while iter in bounds:
      echo(" >  ", iter[])
      # This uses `[]`, not `[]=`:
      iter[] += x.int64
      # This also works, using `[]=`:
      #iter[] = iter[] + x.int64
      inc(iter)
  elif dt == np_int32:
    echo("\narr.dtype is ", $dt)
    echo("iterate through arr elems:")
    var iter = arr.iterateFlat(int32)
    let bounds = arr.getBounds(int32)
    while iter in bounds:
      echo(" >  ", iter[])
      # This uses `[]`, not `[]=`:
      iter[] += x.int32
      # This also works, using `[]=`:
      #iter[] = iter[] + x.int32
      inc(iter)

    #echo("OK, it's finished.  What happens if we try to dereference once more?")
    #echo(" !! ", iter[])  # RangeError (Nim) or IndexError (Python) raised here!
    #
    #   OK, it's finished.  What happens if we try to dereference once more?
    #   Traceback (most recent call last):
    #     File "test_testpymod3.py", line 17, in <module>
    #       testpymod3.myNumpyAdd(b, 5)
    #   IndexError: PyArrayForwardIter[int32] dereferenced at pos 0x2324470,
    #   out of bounds [0x2324420, 0x232446c], with sizeof(int32) == 4
    #   Nim traceback (most recent call last):
    #     File "pmgentestpymod3_wrap.nim", line 116, in exportpy_myNumpyAdd
    #     File "testPymod.nim", line 171, in myNumpyAdd
    #     File "pyarrayiterators.nim", line 112, in []
    #     File "pyarrayiterators.nim", line 102, in assertValid
    #
  else:
    echo("\narr.dtype is ", $dt)
    echo("We won't iterate through this data type today.")

  result = arr


when isMainModule:
  myfunc3(4, "hello", 5.5)
  discard otherfunc3(4, nil)


# This is a function that does not have the `exportpy` pragma.
proc unexportedFunc(d: float64) =
  echo($d)

unexportedFunc(7.7)
# It is prohibited to list unexported functions in the module methods:
#initPyModule("nimtest1", myfunc1, otherfunc1, unexportedFunc)

#dumpTree:
#  proc cstringFunc(s: cstring, i: int): cstring {. exportc .} =
#    echo($s & $i)
#    result = s


# Because several other Nim modules currently #include "numpyarrayobject.h"
# (oops), we need to #include "numpyarrayobject.h" in the generated C-file
# for the Python module too, just so there's a C-file compiled that *doesn't*
# have `NO_IMPORT_ARRAY` defined.
#
# Otherwise, attempting to import the Python module fails with the error:
#  ImportError: ./testpymod1.so: undefined symbol: pymod_ARRAY_API
# which I traced back to this cause.

# TODO: Implement the macro `enablePymodExtension`, when "pymod-extensions.cfg"
# is also implemented (to enable customisation/definition of extensions).
# For now, we will always just enable Numpy by default.
# When this is all implemented, delete the commented-out definitions of the
# macros `initPyModuleExtra` and `initNumpyModule` in the Nim source files
# "pymod.nim" and "pymodpkg/private/includes/realmacrodefs.nim".
#enablePymodExtension("Numpy")


# It's possible to generate multiple (differently-named) Python modules
# from a single annotated Nim module.
initPyModule("testpymod1", myfunc1, otherfunc1, simpleAdd1)
initPyModule("testpymod2", myfunc2, otherfunc2, simpleAdd2)
initPyModule("testpymod3", myfunc3, otherfunc3, myNumpyAdd)

