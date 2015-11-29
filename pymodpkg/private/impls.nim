# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

#
#=== Internal constants
#

const exportpy_nim_wrapper_template = "exportpy_$1"
const exportpy_c_func_name_template = "py_$1"

# We start the template with a non-empty prefix ("pmgen", in this case) to
# allow the user to specify a target Python module filename that begins with
# an underscore ('_').  [Beginning a Python module filename with an underscore
# is one of Tom's preferred practices, but Nim doesn't allow module names to
# begin with underscores.]  Also, it seems that Nim doesn't allow module names
# to contain double-underscores.  Hence, we use the prefix "pmgen" rather than
# the nicer "pmgen_".
const pymod_nim_mod_fname_template = "pmgen$1_wrap.$2"  # Don't include ".nim"
const pymod_c_mod_fname_template = "pmgen$1_capi.c"


import hashes
import macros  # `lineinfo`
#import parsecfg  # Can't seem to use this at compile-time
import strutils  # `normalize`, `cmpIgnoreStyle`, `%`

import pymodpkg/docstrings

import pymodpkg/private/astutils
import pymodpkg/private/registrytypes

import sequtils


#const IntegerSizes = [1, 2, 4, 8]  # the result of sizeof
#
#type CTypeMappingTuple = tuple[
#    pyarg_fmt_str: string,
#    nim_ctype: string,
#    ctype: string,
#    numpy_kind: char,
#    numpy_itemsize: int,
#]
#
#
#proc expectArrayOfCTypeMappingTupleInit(n: NimNode) {. compileTime .} =
#  #hint(treeRepr(n))
#  expectKind(n, nnkStmtList)
#  for i in 0.. <n.len:
#    let elem = n[i]
#    #hint(treeRepr(elem))
#    expectKind(elem, nnkPar)
#    if elem.len != 3:
#      let msg = "expected an array of length 3, got $1 [$2]: " %
#          [$elem.len, elem.lineinfo]
#      error(msg & treeRepr(elem))
#
#    #hint(treeRepr(elem))
#    expectKind(elem[0], nnkCharLit)  # `PyArg_ParseTuple` format string
#    expectKind(elem[1], nnkIdent)  # Nim C-type alias (an identifier)
#    expectKind(elem[2], nnkStrLit)  # C type
#
#
#macro initCTypeMappings(varName: expr, tupleInits: stmt): stmt
#    {. immediate .} =
#  expectArrayOfCTypeMappingTupleInit(tupleInits)
#
#
#initCTypeMappings("CTypeMappings"):
#  # This mapping of `PyArg_ParseTuple` format strings to C types
#  # is from:  https://docs.python.org/2/c-api/arg.html
#  #
#  # The mapping from C types to Nim C-type aliases was performed
#  # manually starting at:  http://nim-lang.org/system.html#clong
#  #
#  # It would probably be more appropriate to order the fields in
#  # the tuple as (format string, C-type string, Nim C-type alias),
#  # since that was the order in which they were worked out (ie, I
#  # ran through the list of format strings on the Python website,
#  # then manually matched them up with the Nim C-type aliases).
#  # I just find this ordering more aesthetically appealing. :P
#  ('s',   cstring,    "const char *")
#  ('b',   cuchar,     "unsigned char")
#  ('h',   cshort,     "short int")
#  ('H',   cushort,    "unsigned short int")
#  ('i',   cint,       "int")
#  ('I',   cuint,      "unsigned int")
#  ('l',   clong,      "long int")
#  ('k',   culong,     "unsigned long")
#  ('L',   clonglong,  "long long")
#  ('K',   culonglong, "unsigned long long")
#  ('n',   csize,      "Py_ssize_t")
#  ('c',   cchar,      "char")
#  ('f',   cfloat,     "float")
#  ('d',   cdouble,    "double")


#
#=== Compile-time utility procs (helpers for macros) for expr & statement manipulation.
#

proc `<<`(ss: var seq[string], s: string) {. compileTime .} =
  ss.add(s)


proc getTargetTypeNodeOfPtr(ptr_node: NimNode, descr: string): NimNode
    {. compileTime .} =
  ## `descr` might be, for example, "proc param" or "return".
  if ptr_node.len == 0:
    let msg = "$1 type is missing the target type of the `ptr` [$2]" %
        [descr, ptr_node.lineinfo]
    error(msg)
  result = ptr_node[0]


proc getPyObjectTypeDef(py_object_type_defs: PyObjectTypeDefTable,
    nim_type_node: NimNode): ref PyObjectTypeDef {. compileTime .} =
  expectKind(nim_type_node, nnkIdent)
  let nim_type = $nim_type_node

  let potd = py_object_type_defs.get(nim_type)
  if potd == nil:
    let msg = "no PyObject type has been defined for Nim type `$1` [$2] (hint: use `definePyObjectType`)" %
        [nim_type, lineinfo(nim_type_node)]
    error(msg)
  return potd  # not nil


proc verifyDefinedPyObjectType(py_object_type_defs: PyObjectTypeDefTable,
    nim_type_node: NimNode): TypeFmtTuple {. compileTime .} =
  let potd = getPyObjectTypeDef(py_object_type_defs, nim_type_node)

  let nim_type = "ptr " & potd.nim_type
  let nim_ctype = "ptr " & potd.nim_type
  let py_type = potd.py_type_label
  var py_fmt_str = if potd.py_obj_ctype == "PyObject": "O" else: "O!"

  return (nim_type, nim_ctype, py_type, py_fmt_str, potd, nil)


proc verifyBuiltinNimType(nim_type_node: NimNode): TypeFmtTuple
    {. compileTime .} =
  #hint("verifyBuiltinNimType: " & treeRepr(nim_type_node))
  expectKind(nim_type_node, nnkIdent)
  let nim_type = $nim_type_node

  # https://docs.python.org/2/c-api/arg.html
  case nim_type
  of "float":
    # NOTE:  The `float` type is defined only as "default floating point type".
    #   -- http://nim-lang.org/docs/system.html#float
    # The Nim Manual elaborates:
    #   "float
    #     the generic floating point type; its size is platform dependent
    #     (the compiler chooses the processor's fastest floating point type).
    #     This type should be used in general."
    #   -- http://nim-lang.org/docs/manual.html#types-pre-defined-floating-point-types
    #
    # So, we'll have to find out whether `float` is `float32` or `float64`...
    if sizeof(float) == sizeof(float32):
      # It appears that `float` is `float32`.
      result = (nim_type, "cfloat", "float", "f", nil, nil)
    else:
      # It appears that `float` is `float64`.
      result = (nim_type, "cdouble", "float", "d", nil, nil)
  of "float32", "cfloat":
    # http://nim-lang.org/system.html#cfloat
    result = (nim_type, "cfloat", "float", "f", nil, nil)
  of "float64", "cdouble":
    # http://nim-lang.org/system.html#cdouble
    result = (nim_type, "cdouble", "float", "d", nil, nil)
  of "int", "int64", "clong":  # 64 bits, signed
    # http://nim-lang.org/system.html#clong
    # NOTE:  The `clong` type is 64 bits, like Nim's `int64` type and the
    # built-in Nim `int` type (on my system, at least), but Python's `int`
    # type is 32 bits, like `cint` (Nim's name for C's `int` type).

    # FIXME: Rather than being hard-coded, this Nim-to-C type mapping
    # should be calculated at compile time, using size-and-signedness
    # matching (to ensure it's correct for the current platform), using
    # the CTypeMappingTuple approach that I that I started (but haven't
    # yet finished) above.
    #
    # For example, Nim's `int` type isn't DEFINED as being 64 bits;
    # it's defined as being the same size as a pointer.  Likewise, `clong`
    # isn't DEFINED as being 64 bits, it just happens to be 64 bits on my
    # system.
    result = (nim_type, "clong", "int", "l", nil, nil)
  of "uint", "uint64", "culong":  # 64 bits, unsigned
    result = (nim_type, "culong", "int", "k", nil, nil)
  of "cint", "int32":  # 32 bits, signed
    result = (nim_type, "cint", "int", "i", nil, nil)
  of "cuint", "uint32":  # 32 bits, unsigned
    result = (nim_type, "cuint", "int", "I", nil, nil)
  of "cshort", "int16":  # 16 bits, signed
    result = (nim_type, "cshort", "int", "h", nil, nil)
  of "cushort", "uint16":  # 16 bits, unsigned
    result = (nim_type, "cushort", "int", "H", nil, nil)

  # What Python format string should we use for this?
  #of "cschar", "int8":  # 8 bits, signed
  #  result = (nim_type, "cschar", "int", "??????", nil, nil)

  of "byte", "uint8":  # 8 bits, unsigned
    result = (nim_type, "byte", "int", "B", nil, nil)
  of "char", "cchar", "cuchar":  # a single string character
    result = (nim_type, "cchar", "str [len == 1]", "c", nil, nil)
  of "string":
    # http://nim-lang.org/system.html#cstring
    result = (nim_type, "cstring", "str", "s", nil, nil)
  else:
    let msg = "unhandled Nim type `$1` [$2]: " % [nim_type, lineinfo(nim_type_node)]
    error(msg & treeRepr(nim_type_node))


proc verifyProcParamType(py_object_type_defs: PyObjectTypeDefTable,
    param_type_node: NimNode): TypeFmtTuple {. compileTime .} =
  #hint("param_type_node = " & treeRepr(param_type_node))
  case param_type_node.kind
  of nnkPtrTy:
    let ptr_target_type_node = getTargetTypeNodeOfPtr(param_type_node, "proc param")
    result = verifyDefinedPyObjectType(py_object_type_defs, ptr_target_type_node)
  else:
    result = verifyBuiltinNimType(param_type_node)


proc verifyProcDef(proc_def_node: NimNode, error_msg: string): string {. compileTime .} =
  expectKind(proc_def_node, nnkProcDef)
  let proc_name_node = proc_def_node.name
  if proc_name_node.kind == nnkEmpty:
    # We can't allow unnamed procs, because we need to be able
    # to call the proc from the C code!
    let msg = error_msg & " [$1]"
    error(msg % lineinfo(proc_def_node))

  # Ensure the proc will be exported from the module; ie, ensure
  # the proc name is followed by an export marker asterisk:
  #  http://nim-lang.org/manual.html#export-marker
  #  http://nim-lang.org/macros.html#postfix-operator-call
  #
  # The AST will look like this:
  #   Hint: ProcDef
  #     Postfix
  #       Ident !"*"
  #       Ident !"yourProcNameHere" [User]
  let proc_name: string = $proc_name_node
  if not proc_name.endsWith("*"):
    # The proc hasn't been exported.  We can't allow unexported procs,
    # because we need to be able to call the procs from the generated
    # module.
    let msg1 = "can't exportpy proc `$1` [$2] because it isn't marked for export from the module."
    let msg2 = "\nHint: Add an asterisk (`*`) after the proc name, as described here:\n $3\n"
    let url = "http://nim-lang.org/manual.html#export-marker"
    let msg = (msg1 & msg2) % [proc_name, lineinfo(proc_def_node), url]
    error(msg)

  let proc_name_without_asterisk = proc_name.substr(0, proc_name.high-1)
  verifyValidCIdent(proc_name_without_asterisk, proc_name_node)
  result = proc_name_without_asterisk


proc verifyProcNameUnique(proc_name: string, proc_def_node: NimNode) {. compileTime .} =
  # Is the identifier-to-be of `proc_name` already in use as the module name?
  # If so, warn now, or it will cause cryptic problems later.
  let li: string = proc_def_node.lineinfo
  let (path_and_filename, mod_name, success) = parseModNameFromLineinfo(li)
  if not success:
    # It didn't work.  Oh well, we were only trying to help.
    return

  if mod_name.normalize == proc_name.normalize:
    let msg = "can't exportpy proc `$1` [$2] that has the same normalized name as its Nim module \"$3\"" %
        [proc_name, li, path_and_filename]
    error(msg)


#
#=== User-invoked macros part 1: Define new PyObject types
#

# This exists for the `PyArg_ParseTuple` format string "O!".
# For more info, see:
#  https://docs.python.org/2/c-api/arg.html
#  https://scipy-lectures.github.io/advanced/interfacing_with_c/interfacing_with_c.html
proc definePyObjectTypeImpl*(
    pyObjectTypeDefs: var PyObjectTypeDefTable,
    nim_type_node,
    py_obj_ctype_node,
    py_type_obj_node,
    py_type_label_node,
    extra_includes_node: NimNode): NimNode {. compileTime .} =
  #hint("nim_type_node = " & treeRepr(nim_type_node))
  expectKind(nim_type_node, nnkIdent)
  expectKind(py_obj_ctype_node, nnkStrLit)
  expectKind(py_type_obj_node, nnkStrLit)
  expectKind(py_type_label_node, nnkStrLit)
  expectArrayOfKind(extra_includes_node, nnkStrLit)

  let nim_type: string = $nim_type_node
  let prev_def_with_this_nim_type = py_object_type_defs.get(nim_type)
  if prev_def_with_this_nim_type != nil:
    let prev_line_info = prev_def_with_this_nim_type.def_line_info
    let msg = "Nim type `$1` [$2] has already been used in a PyObject type def [previously at $3]" %
        [nim_type, lineinfo(nim_type_node), prev_line_info]
    error(msg)

  let py_obj_ctype: string = $py_obj_ctype_node
  verifyValidCIdent(py_obj_ctype, py_obj_ctype_node)
  let py_type_obj: string = $py_type_obj_node
  # For PyObject only, an empty string is allowed (and in fact, expected).
  if nim_type != "PyObject":
    verifyValidCIdent(py_type_obj, py_type_obj_node)
  else:
    if py_type_obj != "":
      let msg = "`PyObject` must be defined with an empty Python type-object string [$1]: " %
          lineinfo(py_type_obj_node)
      error(msg & py_type_obj)

  let py_type_label = $py_type_label_node
  verifyValidPyTypeLabel(py_type_label, py_type_label_node)

  #hint(treeRepr(extra_includes_node))
  let num_extra_includes = extra_includes_node.len
  var extra_includes: seq[string]
  newSeq(extra_includes, num_extra_includes)
  for i in 0.. <num_extra_includes:
    let extra_incl_node = extra_includes_node[i]
    let extra_incl = verifyValidCInclude($extra_incl_node, extra_incl_node)
    extra_includes[i] = extra_incl

  let new_potd = new_PyObjectTypeDef(
      nim_type,
      nim_type_node.lineinfo,
      py_obj_ctype,
      py_type_obj,
      py_type_label,
      extra_includes
  )
  py_object_type_defs << new_potd

  # There's nothing we actually have to write back out...
  result = newStmtList()


proc getTupleReturnType(py_object_type_defs: PyObjectTypeDefTable, return_type_node: NimNode): seq[TypeFmtTuple] {. compileTime .} =
  assert(return_type_node.kind == nnkTupleTy)

  var length = 0
  var return_types = newSeq[TypeFmtTuple](return_type_node.len)
  var return_counts = newSeqWith(return_type_node.len, 0)
  for i in 0 .. <return_type_node.len:
    let node = return_type_node[i]
    for j in 0 .. <node.len:
      if node[j].kind == nnkIdent:
        if j == node.len-1 or node[j+1].kind == nnkEmpty:
          return_types[i] = verifyBuiltinNimType(node[j])
        else:
          inc(length)
          inc(return_counts[i])
      elif node[j].kind == nnkPtrTy:
        let ptr_target_type_node = getTargetTypeNodeOfPtr(node[j], "return")
        return_types[i] = verifyDefinedPyObjectType(py_object_type_defs, ptr_target_type_node)

      elif node[j].kind == nnkTupleTy:
        let msg = "nested tuple return value not implemented [$1]: " % lineinfo(node[j])
        error(msg & treeRepr(node[j]))

  result = newSeq[TypeFmtTuple](length)

  var id = 0
  for i in 0 .. <return_type_node.len:
    let node = return_type_node[i]

    for j in 0 .. <return_counts[i]:
      result[id] = return_types[i]
      result[id].label = $(node[j])
      inc(id)

#
#=== User-invoked macros part 2: exporting Nim procs to Python
#

proc getReturnType(py_object_type_defs: PyObjectTypeDefTable,
    return_type_node: NimNode): seq[TypeFmtTuple] {. compileTime .} =
  case return_type_node.kind
  of nnkEmpty:
    #hint("no return type")
    result = newSeq[TypeFmtTuple](1)
    result[0] = ("void", "void", "None", "", nil, nil)
  of nnkPtrTy:
    #hint("return type: " & treeRepr(return_type_node))
    let ptr_target_type_node = getTargetTypeNodeOfPtr(return_type_node, "return")
    result = @[ verifyDefinedPyObjectType(py_object_type_defs, ptr_target_type_node) ]

  of nnkTupleTy:
    result = getTupleReturnType(py_object_type_defs, return_type_node)

  else:
    #hint("return type: " & treeRepr(return_type_node))
    result = @[ verifyBuiltinNimType(return_type_node) ]


proc countNumIdentsToDefine(param_node: NimNode): int {. compileTime .} =
  expectKind(param_node, nnkIdentDefs)
  result = 0

  #hint("param_node = " & treeRepr(param_node))
  # Some example treeReprs...
  # A parameter that is a pointer:
  #   param_node = IdentDefs
  #     Ident !"pyarr"
  #     PtrTy
  #       Ident !"PyArrayObject"
  #     Empty
  #
  # A parameter that is an int16:
  #   param_node = IdentDefs
  #     Ident !"val"
  #     Ident !"int16"
  #     Empty
  #
  # Three parameters, defined `val1, val2, val3: int16`:
  #   param_node = IdentDefs
  #     Ident !"val1"
  #     Ident !"val2"
  #     Ident !"val3"
  #     Ident !"int16"
  #     Empty

  type PrevStates = enum
    Start, WasIdent, WasEmpty, WasPtr
  var prev_state: PrevStates = Start

  let empty_or_ident_or_ptr_kind = {nnkEmpty, nnkIdent, nnkPtrTy}
  let num_children = param_node.len
  for i in 0.. <num_children:
    let n = param_node[i]
    #hint(treeRepr(n))

    case prev_state
    of Start:
      expectKind(n, nnkIdent)
      inc(result)
      prev_state = WasIdent
    of WasIdent:
      let n_kind = n.kind
      case n_kind
      of nnkIdent:
        # This second ident might be the not-yet-bound identifier for
        # the parameter type.  It's not one of the identifiers we want
        # to count.  (When this proc was written, types were appearing
        # in the AST as symbols rather than identifiers.)  Or it might
        # be another parameter name, separated from the previous param
        # by a comma with no type in-between.
        #
        # How we're going to handle this:  We'll increment the count
        # for each ident we find, then decrement the count when we
        # reach the final empty.
        inc(result)
      of nnkPtrTy:
        # A pointer.  This means that all previous idents were clearly
        # param names; and this is the first element of the type decl.
        # We'll also increment the count, since we know we're going to
        # decrement the count when we reach the empty anyway.
        let ptr_target_type_node = getTargetTypeNodeOfPtr(n, "proc param")
        expectKind(ptr_target_type_node, nnkIdent)
        inc(result)
        prev_state = WasPtr
      of nnkEmpty:
        # OK, so this is the ultimate empty we expected; decrement the
        # count.
        dec(result)
        prev_state = WasEmpty

      of nnkIntLit, nnkFloatLit, nnkStrLit:
        dec(result)
        prev_state = WasEmpty

      else:
        expectKind(n, empty_or_ident_or_ptr_kind)
    of WasEmpty:
      # Why is there something after the empty??
      let msg = "unexpected second Empty node in proc param IdentDefs node [$1]: " %
          lineinfo(param_node)
      error(msg & treeRepr(param_node))
    of WasPtr:
      let n_kind = n.kind
      case n_kind
      of nnkEmpty:
        # A ptr node is unambiguously part of a type declaration, so
        # this empty node is the only valid successor to a ptr node.
        # Decrement the count.
        dec(result)
        prev_state = WasEmpty
      else:
        expectKind(n, nnkEmpty)

  if prev_state != WasEmpty:
    let msg = "incomplete proc param IdentDefs node [$1]: " % lineinfo(param_node)
    error(msg & treeRepr(param_node))


proc getDefaultValue(proc_params: NimNode) : string {. compileTime .} =
    let potential_default = proc_params[proc_params.len-1]

    case potential_default.kind
    of nnkIntLit:
        result = $(potential_default.intVal)
    of nnkFloatLit:
        result = $(potential_default.floatVal)
    of nnkStrLit:
        result = "\"" & potential_default.strVal & "\""
    else:
        result = nil


proc hasPragma(proc_def_node: NimNode; pragma_name: string): bool {. compileTime .} =
  for i in 0 .. <proc_def_node.len:
    let node = proc_def_node[i]
    if node.kind == nnkPragma:
      for j in 0 .. <node.len:
        if cmpIgnoreStyle($(node[j]), pragma_name) == 0:
          return true
  return false


proc exportpyImpl*(
    pyObjectTypeDefs: PyObjectTypeDefTable,
    procPrototypes: var ProcPrototypeTable,
    proc_def_node: NimNode): NimNode {. compileTime .} =

  let proc_name = verifyProcDef(proc_def_node, "can't exportpy unnamed proc")
  #hint("proc name: " & proc_name)
  verifyProcNameUnique(proc_name, proc_def_node)

  let proc_params = params(proc_def_node)
  #hint(treeRepr(proc_params))
  let return_type_node = proc_params[0]  # This will always exist, even if Empty.
  let return_type_fmt_tuple = getReturnType(pyObjectTypeDefs, return_type_node)

  let return_dict = proc_def_node.hasPragma("returnDict")

  # NOTE:  We expect that each `param_node` is of kind `nnkIdentDefs`:
  # it defines an identifier as a parameter-name with a type.  However,
  # there can be more than one identifier defined in a single `param_node`.
  #
  # For example, if the function parameter list is:
  #   (x: int, y: string, z: float64)
  # then there will be 3 IdentDefs nodes, each defining a single identifier:
  #   pymod.nim(269, 12) Hint: arg 1 (len = 3): IdentDefs
  #     Ident !"x"
  #     Sym "int"
  #     Empty [User]
  #   pymod.nim(269, 12) Hint: arg 2 (len = 3): IdentDefs
  #     Ident !"y"
  #     Sym "string"
  #     Empty [User]
  #   pymod.nim(269, 12) Hint: arg 3 (len = 3): IdentDefs
  #     Ident !"z"
  #     Sym "float64"
  #     Empty [User]
  #
  # On the other hand, if the function parameter list is:
  #   (x, y, z: int)
  # then there will be 1 IdentDefs node, which defines 3 identifiers:
  #   pymod.nim(269, 12) Hint: arg 1 (len = 5): IdentDefs
  #     Ident !"x"
  #     Ident !"y"
  #     Ident !"z"
  #     Sym "int"
  #     Empty [User]
  #
  # Hence, we'll process the `proc_params` in 2 passes:
  #  1. count the number of identifiers to be defined
  # (then allocate name+type sequences of the apppropriate length)
  #  2. fill up the sequences with the names+types
  var num_idents = 0
  let num_params = proc_params.len - 1
  var num_idents_per_param: seq[int]
  newSeq(num_idents_per_param, num_params)

  for i in 1..num_params:  # start at 1 because params[0] is the return type.
    let param_node = proc_params[i]
    let j = i-1  # because `i` is counting over range [1, N]
    #hint("arg " & $i & " (len = " & $param_node.len & "): " & treeRepr(param_node))

    let num_idents_in_param_node = countNumIdentsToDefine(param_node)
    num_idents_per_param[j] = num_idents_in_param_node
    num_idents += num_idents_in_param_node
    #hint("num_idents = " & $num_idents)

  # This is for C code generation: variable decls & param names in the kwlist.
  var param_name_type_tuple_seq: seq[ref ParamNameTypeTuple]
  newSeq(param_name_type_tuple_seq, num_idents)

  var storage_idx = 0
  for i in 1..num_params:  # start at 1 because params[0] is the return type.
    let param_node = proc_params[i]
    let j = i-1  # because `i` is counting over range [1, N]
    #hint("arg " & $i & " (len = " & $param_node.len & "): " & treeRepr(param_node))

    let num_idents_in_param_node = num_idents_per_param[j]
    var param_type_node = param_node[num_idents_in_param_node]
    let verified_param_type = verifyProcParamType(py_object_type_defs, param_type_node)

    for k in 0.. <num_idents_in_param_node:
      let name_node = param_node[k]
      expectKind(name_node, nnkIdent)
      let param_name = $name_node
      verifyValidCIdent(param_name, name_node)

      param_name_type_tuple_seq[storage_idx] =
          new_ParamNameTypeTuple(param_name, verified_param_type, getDefaultValue(param_node))

      inc(storage_idx)

  # Write this Nim proc back out (so we can call it eventually).
  # (We should do this, whether or not the proc export succeeds.)
  result = newStmtList()
  result.add(proc_def_node)

  let prev_proc_with_this_name = proc_prototypes.get(proc_name)
  if prev_proc_with_this_name != nil:
    let prev_line_info = prev_proc_with_this_name.proc_line_info
    let msg = "proc name `$1` [$2] has already been exportpy-ed [previously at $3]" %
        [proc_name, lineinfo(proc_def_node), prev_line_info]
    error(msg)

  let docstrings = extractAnyDocstrings(proc_def_node)
  let docstring_lines = splitDocstringLines(docstrings)

  let new_pp = new_ProcPrototype(
      proc_name,
      proc_def_node.lineinfo,
      return_type_fmt_tuple,
      param_name_type_tuple_seq,
      docstring_lines,
      return_dict
  )
  proc_prototypes << new_pp
  #let wrapper_node = generateNimWrapper(new_pp)
  #result.add(wrapper_node)


#
#=== User-invoked macros part 3: Python C-API code generation
#

# FIXME: This should be replaced with the CTypeMappingTuple approach above.
proc convertFormatStringToCType(s: string, n: NimNode): string {. compileTime .} =
  # The branches in this case-statement were copied (even retaining the order
  # of listing) from:
  #  https://docs.python.org/2/c-api/arg.html
  case $s
  of "s":
    result = "const char *"
  of "b":
    result = "unsigned char"
  of "h":
    result = "short int"
  of "H":
    result = "unsigned short int"
  of "i":
    result = "int"
  of "I":
    result = "unsigned int"
  of "l":
    result = "long int"
  of "k":
    result = "unsigned long"
  of "L":
    result = "long long"
  of "K":
    result = "unsigned long long"
  of "n":
    result = "Py_ssize_t"
  of "c":
    result = "char"
  of "f":
    result = "float"
  of "d":
    result = "double"
  else:
    let msg = "unhandled format string [$1]: " % lineinfo(n)
    error(msg & s)


proc extendWithExtraIncludes(output_lines: var seq[string],
    extra_includes_node: NimNode) {. compileTime .} =
  expectArrayOfKind(extra_includes_node, nnkStrLit)
  output_lines << "#include <Python.h>"
  for i in 0.. <extra_includes_node.len:
    let n = extra_includes_node[i]
    let s = verifyValidCInclude($n, n)
    output_lines << "#include " & s


proc createPyArgKeywordList(param_name_type_tuple_seq: seq[ref ParamNameTypeTuple]):
    string {. compileTime .} =
  let num_params = param_name_type_tuple_seq.len
  var quoted_pn_seq: seq[string]
  newSeq(quoted_pn_seq, num_params)
  for i in 0.. <num_params:
    let pn = param_name_type_tuple_seq[i].name
    quoted_pn_seq[i] = "\"$1\", " % pn

  result = "{ $1NULL }" % quoted_pn_seq.join("")


proc generateSafeVariableName(name, proc_name: string): string
    {. compileTime .} =
  # We've already verified that `name` looks like a valid C identifier
  # (using proc `verifyValidCIdent`, called from proc `exportpyImpl`).
  # However, it might overlap with a C keyword or some other identifier.
  # Hence, mangle the name in a predictable fashion.
  #  http://nim-lang.org/hashes.html
  var h: Hash = 0
  h = h !& hash(name)
  h = h !& hash(proc_name)
  h = !$h
  if h < 0:
    h = -h

  result = "$1_$2" % [name, substr($h, 0, 2)]


proc extendWithLocalVars(output_lines: var seq[string],
    nim_wrapper_proc_args_str: var string,
    param_name_type_tuple_seq: seq[ref ParamNameTypeTuple],
    proc_name: string, proc_name_node: NimNode): string
    {. compileTime .} =
  let num_params = param_name_type_tuple_seq.len
  var nim_wrapper_proc_arg_seq: seq[string]
  newSeq(nim_wrapper_proc_arg_seq, num_params)
  var take_addr_of_local_var_seq: seq[string]
  newSeq(take_addr_of_local_var_seq, num_params)
  for i in 0.. <num_params:
    let (param_name, type_fmt_tuple, default_value) = param_name_type_tuple_seq[i][]
    let safe_var_name = generateSafeVariableName(param_name, proc_name)
    nim_wrapper_proc_arg_seq[i] = safe_var_name

    var ctype_str: string
    if type_fmt_tuple.py_object_type_def != nil:
      # It's a defined Python type (eg, Numpy array).
      let potd = type_fmt_tuple.py_object_type_def
      # It will have a pointer sigil.
      ctype_str = "$1 *" % potd.py_obj_ctype

      let py_type_obj = potd.py_type_obj
      if py_type_obj == "":
        # There is no Python type-object to use for type-verification
        # of the PyObject received from the client code.  It will just
        # be left as PyObject.
        take_addr_of_local_var_seq[i] = "&$1" % safe_var_name
      else:
        # We pass an extra Python type-object, to verify the type of
        # the PyObject received from the client code.
        # cf, https://scipy-lectures.github.io/advanced/interfacing_with_c/interfacing_with_c.html#numpy-support
        take_addr_of_local_var_seq[i] = "&$1, &$2" % [py_type_obj, safe_var_name]
    else:
      # It's a built-in Python type.
      let py_fmt_str = type_fmt_tuple.py_fmt_str
      ctype_str = convertFormatStringToCType(py_fmt_str, proc_name_node)
      take_addr_of_local_var_seq[i] = "&$1" % safe_var_name

    var default_init = if default_value == nil: "" else: " = " & default_value;

    var padding = " "
    if ctype_str.endsWith("*"):
      # Right-align C pointers against their variable names (no padding).
      # We use this for both "PyObject *" and "const char *".
      padding = ""
    let var_decl = "\t$1$2$3$4;" % [ctype_str, padding, safe_var_name, default_init]
    output_lines << var_decl

  nim_wrapper_proc_args_str = nim_wrapper_proc_arg_seq.join(", ")
  result = take_addr_of_local_var_seq.join(", ")


proc extendWithOneFunctionDef(output_lines: var seq[string],
    pp: ref ProcPrototype, proc_name: string, proc_name_node: NimNode)
    {. compileTime .} =
  output_lines << ""
  output_lines << "/*"
  output_lines << " * Auto-generated from exported function `$1`:" % proc_name
  output_lines << " *  $1" % pp.proc_line_info
  output_lines << " */"
  output_lines << "static PyObject *"

  let c_func_name = exportpy_c_func_name_template % proc_name
  let c_func_prototype = c_func_name & "(PyObject *class_, PyObject *args, PyObject *kwargs)"
  output_lines << c_func_prototype
  output_lines << "{"

  let nim_wrapper_proc_name = exportpy_nim_wrapper_template % proc_name
  var nim_wrapper_proc_args = ""

  let params = pp.param_name_type_tuple_seq
  let num_params = params.len
  if num_params > 0:
    let take_addrs_of_local_vars_str =
        extendWithLocalVars(output_lines, nim_wrapper_proc_args,
            params, proc_name, proc_name_node)
    output_lines << ""

    let kw_list_definition = "\tstatic char *kwlist[] = $1;" %
        createPyArgKeywordList(params)
    output_lines << kw_list_definition

    var param_type_fmt_seq: seq[string]
    newSeq(param_type_fmt_seq, num_params+1)
    var j = 0
    for i in 0.. <num_params:
      let param = params[i]
      if i == j and param.default_value != nil:
          param_type_fmt_seq[j] = "|"
          inc(j)
      param_type_fmt_seq[j] = param.type_fmt_tuple.py_fmt_str
      inc(j)
    if j == num_params:
        param_type_fmt_seq[num_params] = ""

    let PyArg_ParseTuple_args = "args, kwargs, \"$1\", kwlist,\n\t\t\t$2" %
        [param_type_fmt_seq.join(""), take_addrs_of_local_vars_str]
    let PyArg_ParseTuple_invoc = "\tif (! PyArg_ParseTupleAndKeywords($1)) {" %
        PyArg_ParseTuple_args
    output_lines << PyArg_ParseTuple_invoc
    output_lines << "\t\treturn NULL;"
    output_lines << "\t}"

  output_lines << ""
  output_lines << "\treturn $1($2);" % [nim_wrapper_proc_name, nim_wrapper_proc_args]
  output_lines << "}"
  output_lines << ""


proc extendWithAllFunctionDefs(output_lines: var seq[string],
    proc_prototypes: ProcPrototypeTable,
    proc_names_node: NimNode, mod_name: string) {. compileTime .} =
  expectArrayOfKind(proc_names_node, nnkSym)
  let num_proc_names = proc_names_node.len
  for i in 0.. <num_proc_names:
    let proc_name_node = proc_names_node[i]
    let proc_name = $proc_name_node
    let pp = proc_prototypes.get(proc_name)
    if pp == nil:
      let msg = "proc `$1` must be exported using \"exportpy\" pragma, before it can be listed in \"$2\" module methods [at $3]" %
          [proc_name, mod_name, lineinfo(proc_name_node)]
      error(msg)

    extendWithOneFunctionDef(output_lines, pp, proc_name, proc_name_node)


template outputPyMethodDefDoc(output_lines: var seq[string], s: string) =
  output_lines << "\t\t\"$1\\n\"" % s


proc py_type(type_fmt_tuples: seq[TypeFmtTuple]; return_dict: bool = false) : string {. compileTime .} =
  if return_dict and type_fmt_tuples[0].label != nil:
    result = "{ $1: $2" % [type_fmt_tuples[0].label, type_fmt_tuples[0].py_type]
    for i in 1 .. <type_fmt_tuples.len:
      result = result & (", $1: $2" % [type_fmt_tuples[i].label, type_fmt_tuples[i].py_type])
    result = result & " }"
  elif type_fmt_tuples[0].nim_type != "void":
    result = "(" & type_fmt_tuples[0].py_type
    for i in 1 .. <type_fmt_tuples.len:
      result = result & ", " & type_fmt_tuples[i].py_type
    result = result & ")"
  else:
    result = "None"

proc py_fmt_str(type_fmt_tuples: seq[TypeFmtTuple]; return_dict: bool = false) : string {. compileTime .} =
  # "O!" is not a valid format string for `Py_BuildValue`.
  # We convert it to O when building the combined format string.
  if return_dict and type_fmt_tuples[0].label != nil:
    result = "{"
    for i in 0 .. <type_fmt_tuples.len:
      let py_fmt_str = type_fmt_tuples[i].py_fmt_str
      result = result & "s" & (if py_fmt_str == "O!": "O" else: py_fmt_str)
    result = result & "}"
  elif type_fmt_tuples[0].nim_type != "void":
    result = ""
    for i in 0 .. <type_fmt_tuples.len:
      let py_fmt_str = type_fmt_tuples[i].py_fmt_str
      result = result & (if py_fmt_str == "O!": "O" else: py_fmt_str)
  else:
    result = ""

proc nim_type(type_fmt_tuples: seq[TypeFmtTuple]) : string {. compileTime .} =
  if type_fmt_tuples[0].nim_type != "void":
    result = "(" & type_fmt_tuples[0].nim_type
    for i in 1 .. <type_fmt_tuples.len:
      result = result & ", " & type_fmt_tuples[i].nim_type
    result = result & ")"
  else:
    result = "void"


proc extendWithPyFuncPrototypeDoc(output_lines: var seq[string],
    pp: ref ProcPrototype, proc_name: string, proc_name_node: NimNode)
    {. compileTime .} =
  # This syntax of type annotations on function prototypes is based upon
  #  https://www.python.org/dev/peps/pep-0484/
  let params = pp.param_name_type_tuple_seq
  let num_params = params.len

  var params_and_types: seq[string]
  newSeq(params_and_types, num_params)
  for i in 0.. <num_params:
    let p = params[i]
    let p_name = p.name
    let py_type = p.type_fmt_tuple.py_type
    params_and_types[i] = "$1: $2" % [p_name, py_type]

  let params_str = params_and_types.join(", ")
  let return_type_str = pp.return_type_fmt_tuple.py_type
  let prototype_str = "$1($2) -> $3" % [proc_name, params_str, return_type_str]
  outputPyMethodDefDoc(output_lines, prototype_str)
  outputPyMethodDefDoc(output_lines, "")


proc extendWithPyFuncParametersDoc(output_lines: var seq[string],
    pp: ref ProcPrototype, proc_name: string, proc_name_node: NimNode)
    {. compileTime .} =
  let params = pp.param_name_type_tuple_seq
  let num_params = params.len
  if num_params > 0:
    outputPyMethodDefDoc(output_lines, "Parameters")
    outputPyMethodDefDoc(output_lines, "----------")

    for i in 0.. <num_params:
      let p = params[i]
      let p_name = p.name
      let py_type = p.type_fmt_tuple.py_type
      let nim_type = p.type_fmt_tuple.nim_type
      let s = "$1 : $2 -> $3" % [p_name, py_type, nim_type]
      outputPyMethodDefDoc(output_lines, s)
    outputPyMethodDefDoc(output_lines, "")

  outputPyMethodDefDoc(output_lines, "Returns")
  outputPyMethodDefDoc(output_lines, "-------")

  let py_type = pp.return_type_fmt_tuple.py_type(pp.return_dict)
  let nim_type = pp.return_type_fmt_tuple.nim_type
  let s = "out : $1 <- $2" % [py_type, nim_type]
  outputPyMethodDefDoc(output_lines, s)
  outputPyMethodDefDoc(output_lines, "")


proc extendWithOnePyMethodDef(output_lines: var seq[string],
    proc_prototypes: ProcPrototypeTable,
    proc_name_node: NimNode, mod_name: string) {. compileTime .} =
  let proc_name = $proc_name_node
  let pp = proc_prototypes.get(proc_name)
  if pp == nil:
    let msg = "proc `$1` must be exported using \"exportpy\" pragma, before it can be listed in \"$2\" module methods [at $3]" %
        [proc_name, mod_name, lineinfo(proc_name_node)]
    error(msg)

  let c_func_name = exportpy_c_func_name_template % proc_name
  let method_line = "\t{ \"$1\", (PyCFunction) $2, METH_VARARGS | METH_KEYWORDS," %
      [proc_name, c_func_name]
  output_lines << method_line
  extendWithPyFuncPrototypeDoc(output_lines, pp, proc_name, proc_name_node)
  extendWithPyFuncParametersDoc(output_lines, pp, proc_name, proc_name_node)
  for s in pp.docstring_lines:
    output_lines << "\t\t\"$1\\n\"" % s.replace("\"", "\\\"")
  output_lines << "\t},"


proc extendWithPyMethodDefs(output_lines: var seq[string],
    proc_prototypes: ProcPrototypeTable,
    proc_names_node: NimNode, mod_name: string) {. compileTime .} =
  output_lines << ""
  output_lines << "static PyMethodDef methods[] = {"

  let num_proc_names = proc_names_node.len
  for i in 0.. <num_proc_names:
    let proc_name_node = proc_names_node[i]
    extendWithOnePyMethodDef(output_lines, proc_prototypes, proc_name_node, mod_name)
  output_lines << "\t{ NULL, NULL, 0, NULL },"
  output_lines << "};"


when defined(python3):
  proc extendWithPyModinitFunc(output_lines: var seq[string],
      extra_init_node: NimNode, mod_name: string) {. compileTime .} =
    output_lines << ""
    output_lines << "/*"
    output_lines << " * This port to Python3 is based upon the example code at:"
    output_lines << " *  https://docs.python.org/3/howto/cporting.html"
    output_lines << " * and:"
    output_lines << " *  http://docs.scipy.org/doc/numpy/user/c-info.ufunc-tutorial.html#example-non-ufunc-extension"
    output_lines << " */"
    output_lines << "struct module_state {"
    output_lines << "\tPyObject *error;"
    output_lines << "};"
    output_lines << ""
    output_lines << "static struct PyModuleDef module_def = {"
    output_lines << "\tPyModuleDef_HEAD_INIT,"
    output_lines << "\t\"$1\",                         /* m_name */" % mod_name
    output_lines << "\t\"\",                             /* m_doc */"
    output_lines << "\tsizeof(struct module_state),    /* m_size */"
    output_lines << "\tmethods,                        /* m_methods */"
    output_lines << "\tNULL,                           /* m_reload */"
    output_lines << "\tNULL,                           /* m_traverse */"
    output_lines << "\tNULL,                           /* m_clear */"
    output_lines << "\tNULL,                           /* m_free */"
    output_lines << "};"
    output_lines << ""
    output_lines << "PyMODINIT_FUNC"
    output_lines << "PyInit_$1(void) {" % mod_name
    output_lines << "\tPyObject *m = PyModule_Create(&module_def);"
    output_lines << "\tif (!m) {"
    output_lines << "\t\treturn NULL;"
    output_lines << "\t}"

    let num_extra_init = extra_init_node.len
    for i in 0.. <num_extra_init:
      let ei = $extra_init_node[i]
      output_lines << "\t$1" % ei
    output_lines << "\tNimMain();"
    output_lines << "\treturn m;"
    output_lines << "}"

else:
  proc extendWithPyModinitFunc(output_lines: var seq[string],
      extra_init_node: NimNode, mod_name: string) {. compileTime .} =
    output_lines << ""
    output_lines << "PyMODINIT_FUNC"
    output_lines << "init$1(void)" % mod_name
    output_lines << "{"
    output_lines << "\tPyObject *m = Py_InitModule(\"$1\", methods);" % mod_name
    output_lines << "\tif (m == NULL) {"
    output_lines << "\t\treturn;"
    output_lines << "\t}"

    let num_extra_init = extra_init_node.len
    for i in 0.. <num_extra_init:
      let ei = $extra_init_node[i]
      output_lines << "\t$1" % ei
    output_lines << "\tNimMain();"
    output_lines << "}"


proc outputPyModuleC(
    proc_prototypes: ProcPrototypeTable,
    mod_name: string,
    extra_includes_node: NimNode, extra_init_node: NimNode,
    proc_names_node: NimNode) {. compileTime .} =
  let c_mod_fname = pymod_c_mod_fname_template % mod_name
  #hint(c_mod_fname)
  let nim_mod_header_fname = pymod_nim_mod_fname_template % [mod_name, "h"]

  # http://nim-lang.org/system.html#CompileDate
  let compilation_date_time = "/* Auto-generated by Pymod on $1 at $2 */" %
      [CompileDate, CompileTime]
  var output_lines: seq[string] = @[compilation_date_time, ""]
  output_lines << "#define YES_IMPORT_ARRAY"
  extendWithExtraIncludes(output_lines, extra_includes_node)
  # TODO: This should actually instead by the header file generated for the
  # exported Nim procs, which will itself #include "nimbase.h"
  output_lines << "#include \"nimcache/$1\"" % nim_mod_header_fname
  output_lines << ""
  extendWithAllFunctionDefs(output_lines, proc_prototypes, proc_names_node, mod_name)
  extendWithPyMethodDefs(output_lines, proc_prototypes, proc_names_node, mod_name)
  output_lines << ""
  extendWithPyModinitFunc(output_lines, extra_init_node, mod_name)

  let output_content = output_lines.join("\n")
  #hint(output_content)
  writeFile(c_mod_fname, output_content)
  hint("Created C file: " & c_mod_fname)


proc extendWithNimProcPrototype(output_lines: var seq[string],
    pp: ref ProcPrototype, proc_name: string, proc_name_node: NimNode)
    {. compileTime .} =
  output_lines << "#"
  output_lines << "# Regardless of the return-type of the underlying Nim proc, we always"
  output_lines << "# return `ptr PyObject`, so we can return `nil` if an exception is raised."

  let params = pp.param_name_type_tuple_seq
  let num_params = params.len

  var params_and_types: seq[string]
  newSeq(params_and_types, num_params)
  for i in 0.. <num_params:
    let p = params[i]
    let p_name = p.name
    let nim_ctype = p.type_fmt_tuple.nim_ctype
    params_and_types[i] = "$1: $2" % [p_name, nim_ctype]

  let nim_wrapper_proc_name = exportpy_nim_wrapper_template % proc_name
  let params_str = params_and_types.join(", ")
  # http://forum.nim-lang.org/t/634
  # http://forum.nim-lang.org/t/573
  const pragma_str = "{. exportc, dynlib, cdecl .}"
  let prototype_str = "proc $1($2): ptr PyObject" %
      [nim_wrapper_proc_name, params_str]
  output_lines << prototype_str
  output_lines << "        $1 =" % pragma_str


const NimWrapperBodyTemplate = """
  # http://nim-lang.org/manual.html#defer-statement
  defer: collectAllGarbage()

  try:
    initRegisteredPyObjects()
    $1
    # $2
    return $3
  except AssertionError:
    let msg = "$$1\n$$2" % [getCurrentExceptionMsg(),
        prettyPrintStackTrace(getStackTrace(getCurrentException()))]
    return raisePyAssertionError(msg)
  except IndexError:
    let msg = "$$1\n$$2" % [getCurrentExceptionMsg(),
        prettyPrintStackTrace(getStackTrace(getCurrentException()))]
    return raisePyIndexError(msg)
  except KeyError:
    let msg = "$$1\n$$2" % [getCurrentExceptionMsg(),
        prettyPrintStackTrace(getStackTrace(getCurrentException()))]
    return raisePyKeyError(msg)
  except ObjectConversionError:
    let msg = "$$1\n$$2" % [getCurrentExceptionMsg(),
        prettyPrintStackTrace(getStackTrace(getCurrentException()))]
    return raisePyTypeError(msg)
  except RangeError:
    let msg = "$$1\n$$2" % [getCurrentExceptionMsg(),
        prettyPrintStackTrace(getStackTrace(getCurrentException()))]
    return raisePyIndexError(msg)  # There's no RangeError in Python.
  except ValueError:
    let msg = "$$1\n$$2" % [getCurrentExceptionMsg(),
        prettyPrintStackTrace(getStackTrace(getCurrentException()))]
    return raisePyValueError(msg)
  except:  # catch any other Exception
    let msg = "$$1\n$$2" % [getCurrentExceptionMsg(),
        prettyPrintStackTrace(getStackTrace(getCurrentException()))]
    return raisePyRuntimeError(msg)
  finally:
    # Anything that needs to run after any exception has been handled.
    discard

  return getPyNone()
"""


proc extendWithOneNimWrapperProcDef(output_lines: var seq[string],
    pp: ref ProcPrototype, proc_name: string, proc_name_node: NimNode)
    {. compileTime .} =

  output_lines << ""
  output_lines << "# Auto-generated from exported function `$1`:" % proc_name
  output_lines << "#  $1" % pp.proc_line_info
  extendWithNimProcPrototype(output_lines, pp, proc_name, proc_name_node)

  let params = pp.param_name_type_tuple_seq
  let num_params = params.len

  var func_args: seq[string]
  newSeq(func_args, num_params)
  for i in 0.. <num_params:
    let p = params[i]
    let p_name = p.name
    let p_type = p.type_fmt_tuple.nim_ctype
    if p_type == "cstring":
      # Convert the cstring to a string.
      #  http://nim-lang.org/manual.html#cstring-type
      func_args[i] = "$" & p_name
    else:
      func_args[i] = p_name

  let return_type = pp.return_type_fmt_tuple.nim_type
  if return_type != "void":
    let func_call = "let return_val = $1($2)" % [proc_name, func_args.join(", ")]
    var return_type_fmt_str = pp.return_type_fmt_tuple.py_fmt_str(pp.return_dict)

    var comment : string
    var return_val : string
    if pp.return_type_fmt_tuple[0].label == nil:
      # Straightforward single return value
      comment = if return_type_fmt_str == "O":
          "Note: `Py_BuildValue(\"O\")` increments the ref-count of the object."
        else:
          "Create a new PyObject value from the Nim value."
      if pp.return_dict:
        comment = comment & " Ignoring \"returnDict\" pragma for non-tuple."

      return_val = "Py_BuildValue(\"$1\", return_val)" % return_type_fmt_str
    elif pp.return_dict:
      # Multiple return values in a dict
      comment = "Construct dict from the Nim return values."
      return_val = "PyBuildValue(\"$1\"" % return_type_fmt_str
      for i in 0 .. <pp.return_type_fmt_tuple.len:
        let label = pp.return_type_fmt_tuple[i].label
        return_val = return_val & (", \"$1\", return_val.$1" % label)
      return_val = return_val & ")"
    else:
      # Multiple return values in a tuple
      comment = "Construct tuple from the Nim return values."
      return_val = "PyBuildValue(\"$1\"" % return_type_fmt_str
      for i in 0 .. <pp.return_type_fmt_tuple.len:
        return_val = return_val & (", return_val.$1" % pp.return_type_fmt_tuple[i].label)
      return_val = return_val & ")"

    output_lines << NimWrapperBodyTemplate % [func_call, comment, return_val]

  else:
    let func_call = "$1($2)" % [proc_name, func_args.join(", ")]
    let comment = "No return value => return None."
    let return_val = "getPyNone()"
    output_lines << NimWrapperBodyTemplate % [func_call, comment, return_val]
  output_lines << ""

proc extendWithAllNimWrapperProcDefs(output_lines: var seq[string],
    proc_prototypes: ProcPrototypeTable,
    proc_names_node: NimNode, mod_name: string) {. compileTime .} =
  expectArrayOfKind(proc_names_node, nnkSym)
  let num_proc_names = proc_names_node.len
  for i in 0.. <num_proc_names:
    let proc_name_node = proc_names_node[i]
    let proc_name = $proc_name_node
    let pp = proc_prototypes.get(proc_name)
    if pp == nil:
      let msg = "proc `$1` must be exported using \"exportpy\" pragma, before it can be listed in \"$2\" module methods [at $3]" %
          [proc_name, mod_name, lineinfo(proc_name_node)]
      error(msg)

    extendWithOneNimWrapperProcDef(output_lines, pp, proc_name, proc_name_node)


proc outputPyModuleNim(
    proc_prototypes: ProcPrototypeTable,
    nimModulesToImport: NimModulesToImportTable,
    mod_name: string,
    proc_names_node: NimNode)
    {. compileTime .} =
  let nim_mod_fname = pymod_nim_mod_fname_template % [mod_name, "nim"]
  #hint(nim_mod_fname)

  # http://nim-lang.org/system.html#CompileDate
  let compilation_date_time = "# Auto-generated by Pymod on $1 at $2" %
      [CompileDate, CompileTime]

  # "Note that you can use gorge from the system module to embed parameters
  # from an external command at compile time":
  #  http://nim-lang.org/nimc.html#passc-pragma

  let c_mod_fname = pymod_c_mod_fname_template % mod_name
  let compile_pragma = "compile: \"$1\"" % c_mod_fname
  let pragma_line = "{. $1 .}" % compile_pragma
  var output_lines: seq[string] = @[
      compilation_date_time,
      "",
      pragma_line,
      ""
  ]
  output_lines << "import strutils"
  output_lines << ""
  output_lines << "import pymodpkg/miscutils"
  output_lines << "import pymodpkg/pyobject"
  output_lines << "import pymodpkg/private/membrain"
  # FIXME:  Ideally, we only want to import `pyarrayobject` if we need to.
  # However, this sin is also made by `definePyObjectType(PyArrayObject,`
  # in "pymod.nim".
  output_lines << "import pymodpkg/pyarrayobject"
  for nm in nimModulesToImport:
    output_lines << "import $1" % nm
  output_lines << ""
  extendWithAllNimWrapperProcDefs(output_lines, proc_prototypes, proc_names_node, mod_name)

  let output_content = output_lines.join("\n")
  #hint(output_content)
  writeFile(nim_mod_fname, output_content)
  hint("Created Nim file: " & nim_mod_fname)


proc outputPyModuleNimCfg(mod_name: string, proc_names_node: NimNode)
    {. compileTime .} =
  let nim_mod_fname = pymod_nim_mod_fname_template % [mod_name, "nim"]
  let output_shared_library_fname = "$1.so" % mod_name
  let out_cfg = "out:\"$1\"" % output_shared_library_fname

  # Read user-specified configuration parameters from a cfg file on disk.
  # http://nim-lang.org/system.html#staticRead
  #let pyobject_cfg = staticRead("PyObject.cfg")
  #hint("PyObject.cfg: " & pyobject_cfg)
  #let pyarrayobject_cfg = staticRead("PyArrayObject.cfg")
  #hint("PyArrayObject.cfg: " & pyarrayobject_cfg)
  #var parser: CfgParser
  # Can't invoke `newStrignStream` at compile time:
  #  lib/pure/lexbase.nim(149, 10) Error: VM is not allowed to 'cast'
  # Line 149 of "lib/pure/lexbase.nim" is:
  #  L.buf = cast[cstring](alloc(bufLen * chrSize))
  # in proc `open`.
  #open(parser, newStringStream("PyObject.cfg"), "PyObject.cfg")
  # Can't invoke `newFileStream` at compile time:
  #  lib/system/sysio.nim(217, 19) Error: cannot 'importc' variable at compile time
  # Line 217 of "lib/system/sysio.nim" is:
  #  var p: pointer = fopen(filename, FormatOpen[mode])
  # in proc `open`.
  #open(parser, newFileStream("PyObject.cfg", fmRead), "PyObject.cfg")

  # OK, I don't have time to implement that.  Just hard-code them for now...
  # FIXME:  Do this properly one day?

  # NOTE:  We want to compile the auto-generated Nim & C source files in a
  # single Nim compiler invocation.  However, the auto-generated C source
  # needs a C header file that will be generated from the auto-generated
  # Nim source during the compiler invocation (ie, this header file doesn't
  # exist BEFORE the compiler invocation.
  #
  # Hence, to compile the auto-generated Nim & C source files successfully,
  # we rely upon the header file being generated before the C source
  # compilation begins.  However, both the generation of the header file
  # and the compilation of the C source are triggered by the compilation of
  # the Nim source:  The generation of the C header file from the Nim source
  # is triggered by the "header" pragma in the "modname.nim.cfg" file, while
  # the compilation of the C source is triggered by the "compile" pragma in
  # the Nim source.
  #
  # Hence, we're relying upon the "modname.nim.cfg" file that corresponds to
  # the Nim source file being processed BEFORE the Nim source itself.
  #
  # NOTE: #2: Don't use the "clib:library" option for this; it doesn't search
  # the C linker library search path for the specified C library; instead, it
  # assumes the C library is in the current directory.
  # Instead, use "passL:-llibrary"; it DOES search the search path properly.

  let nim_mod_cfg_content = [
      "app:lib",
      #"clib:\"python2.7\"",
      "header",
      "noMain",
      out_cfg,
      #"passL:\"-lpython2.7 -fPIC\"",
      ""  # Join an empty string, so the content ends with a newline.
  ].join("\n")
  let nim_mod_cfg_fname = "$1.cfg" % nim_mod_fname
  writeFile(nim_mod_cfg_fname, nim_mod_cfg_content)
  hint("Created Nim cfg file: " & nim_mod_cfg_fname)


proc initPyModuleImpl*(
    pyObjectTypeDefs: PyObjectTypeDefTable,
    procPrototypes: ProcPrototypeTable,
    nimModulesToImport: NimModulesToImportTable,
    mod_name_node: NimNode,
    extra_includes_node: NimNode,
    extra_init_node: NimNode,
    proc_names_node: NimNode): NimNode {. compileTime .} =

  expectKind(mod_name_node, nnkStrLit)
  expectArrayOfKind(extra_includes_node, nnkStrLit)
  expectArrayOfKind(extra_init_node, nnkStrLit)
  expectArrayOfKind(proc_names_node, nnkSym)

  var mod_name: string = $mod_name_node
  if mod_name.len == 0:
    # Default to "_%(nim_mod_name)s".
    mod_name = "_" & mod_name_node.getModuleName

  #hint("mod name: " & mod_name)
  verifyValidCIdent(mod_name, mod_name_node)
  outputPyModuleC(procPrototypes, mod_name, extra_includes_node, extra_init_node, proc_names_node)
  outputPyModuleNim(procPrototypes, nimModulesToImport, mod_name, proc_names_node)
  outputPyModuleNimCfg(mod_name, proc_names_node)

  result = newStmtList()

