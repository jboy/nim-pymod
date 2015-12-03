# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import hashes
import strutils
#import tables  # Can't seem to use this at compile-time


#
#=== Registries of exported procs & defined PyObject types
#

# This exists for the `PyArg_ParseTuple` format string "O!".
# For more info, see:
#  https://docs.python.org/2/c-api/arg.html
#  https://scipy-lectures.github.io/advanced/interfacing_with_c/interfacing_with_c.html
type PyObjectTypeDef* = tuple[
    # The type defined to be used in Nim, eg "PyArrayObject".
    # Note that it DOESN'T include the "ptr" prefix (even though
    # it will always appear with a "ptr" prefix in generated code).
    # This will be used in Nim files (both original & auto-generated).
    nim_type: string,
    # The Nim source line on which the PyObjectType was defined.
    def_line_info: string,
    # The type in C of the PyObject substitute, eg "PyArrayObject".
    # Note that it DOESN'T include the "*" pointer sigil (even though
    # it will always appear with a pointer sigil in generated code).
    # This will be used in the generated C files.
    py_obj_ctype: string,
    # The Python type-object that represents the PyObject substitute,
    # eg "PyArray_Type".  This is used by `PyArg_ParseTuple` to verify
    # the type of the PyObject received from the client code.
    # cf, https://scipy-lectures.github.io/advanced/interfacing_with_c/interfacing_with_c.html#numpy-support
    # This will be used in the generated C files.
    py_type_obj: string,
    # A human-readable label for the Python type, eg "numpy.ndarray".
    # This is just used in the generated docstring for Python function.
    py_type_label: string,
    # Any extra #includes needed, eg ["<numpy/arrayobject.h>"].
    extra_includes: seq[string]
]

proc new_PyObjectTypeDef*(
    nim_type: string,
    def_line_info: string,
    py_obj_ctype: string,
    py_type_obj: string,
    py_type_label: string,
    extra_includes: seq[string]): ref PyObjectTypeDef {. compileTime .} =
  new(result)

  result.nim_type = nim_type
  result.def_line_info = def_line_info
  result.py_obj_ctype = py_obj_ctype
  result.py_type_obj = py_type_obj
  result.py_type_label = py_type_label
  result.extra_includes = extra_includes

proc getKey*(potd: ref PyObjectTypeDef): string {. compileTime .} =
  result = potd.nim_type


type TypeFmtTuple* = tuple[
    # The Nim type of the parameter or return value being exported.
    # Note that it DOES contain any "ptr" prefix that will be used
    # a PyObject type definition.
    nim_type: string,
    # The corresponding "c"-type in Nim; eg, "cint", "cstring".
    # Note that it DOES contain any "ptr" prefix that will be used
    # a PyObject type definition.
    nim_ctype: string,
    # The type in Python, whether built-in or defined by a library.
    # This is just used in the generated docstring for Python function.
    py_type: string,
    # The `PyArg_ParseTuple` format string.
    py_fmt_str: string,
    # A reference to the PyObject type definition for this Nim type,
    # if this is a PyObject type definition; otherwise, nil.
    py_object_type_def: ref PyObjectTypeDef,

    # Optional, a label for the return value (used for tuples)
    label: string
]

type ParamNameTypeTuple* = tuple[
    name: string,
    type_fmt_tuple: TypeFmtTuple,
    default_value: string
]

proc new_ParamNameTypeTuple*(
    name: string,
    type_fmt_tuple: TypeFmtTuple,
    default_value: string):
    ref ParamNameTypeTuple {. compileTime .} =
  new(result)

  result.name = name
  result.type_fmt_tuple = type_fmt_tuple
  result.default_value = default_value


type ProcPrototype* = tuple[
    proc_name: string,
    proc_line_info: string,
    return_type_fmt_tuple: seq[TypeFmtTuple],
    param_name_type_tuple_seq: seq[ref ParamNameTypeTuple],
    docstring_lines: seq[string],
    do_return_dict: bool
]

proc new_ProcPrototype*(
    proc_name: string,
    proc_line_info: string,
    return_type_fmt_tuple: seq[TypeFmtTuple],
    param_name_type_tuple_seq: seq[ref ParamNameTypeTuple],
    docstring_lines: seq[string],
    do_return_dict: bool = false):
    ref ProcPrototype {. compileTime .} =
  new(result)

  result.proc_name = proc_name
  result.proc_line_info = proc_line_info
  result.return_type_fmt_tuple = return_type_fmt_tuple
  result.param_name_type_tuple_seq = param_name_type_tuple_seq
  result.docstring_lines = docstring_lines
  result.do_return_dict = do_return_dict

proc getKey*(ptfs: ref ProcPrototype): string {. compileTime .} =
  result = ptfs.proc_name


# Implementation detail:
#
# I originally tried to use a `TableRef[string, ProcPrototype]` (from the
# `tables` module: http://nim-lang.org/tables.html ) for this compile-time
# collection, to avoid the increasing O(N) costs of appending to a resizing
# contiguous-memory sequence and searching linearly through a sequence.
#
# But when I tried to put a ProcPrototype instance into the TableRef
# instance (even after the TableRef instance had been initialised using the
# var-definition commented-out at the end of this comment), the compilation
# failed with:
#
#   SIGSEGV: Illegal storage access.
#   (Try to compile with -d:useSysAssert -d:useGcAssert for details.)
#
#var proc_prototypes: TableRef[string, ProcPrototype] =
#        newTable[string, ProcPrototype](128)
#
# Hence, we'll have to make do with the built-in sequence type.


# Hash `proc_name` for more-efficient comparison during iterations.
type HashedElem[T] = tuple[hashedKey: Hash, storedVal: ref T]


proc get*[T](tab: seq[HashedElem[T]], key: string): ref T {. compileTime .} =
  let h = hash(key)
  for e in items(tab):
    if e.hashedKey == h:
      if getKey(e.storedVal) == key:
        return e.storedVal
  return nil


proc `<<`*[T](tab: var seq[HashedElem[T]], val: ref T) {. compileTime .} =
  # Avoid Nim compiler bug: https://github.com/Araq/Nim/issues/2369
  #var h: HashedElem[T] = (hashedKey: hash(getKey(val)), storedVal: val)
  #tab.add(h)
  tab.add((hashedKey: hash(getKey(val)), storedVal: val))


type PyObjectTypeDefTable* = seq[HashedElem[PyObjectTypeDef]]
type ProcPrototypeTable* = seq[HashedElem[ProcPrototype]]
type NimModulesToImportTable* = seq[string]

