# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import macros  # `lineinfo`
import os  # `splitFile`
import strutils  # `%`, `IdentChars`, `IdentStartChars`, `find`, `substr`


proc createStrLitArray*(elems: varargs[string]): NimNode {. compileTime .} =
  result = newNimNode(nnkBracket)
  for i in 0.. <elems.len:
    result.add(newStrLitNode(elems[i]))


proc expectArrayOfKind*(n: NimNode, k: NimNodeKind)
    {. compileTime, raises: [], tags: [] .} =
  #hint(treeRepr(n))
  expectKind(n, nnkBracket)
  for i in 0.. <n.len:
    let elem = n[i]
    #hint(treeRepr(elem))
    expectKind(elem, k)


proc parseModNameFromLineinfo*(li: string):
    tuple[path_and_filename, mod_name: string; success: bool] {. compileTime .} =
  # FIXME:  Is there a better way to accomplish this?  ie, a stdlib Nim proc?
  result = (nil, nil, false)

  # The NimNode.lineinfo string takes the form "path/to/filename(line, col)".
  #  -- http://nim-lang.org/docs/macros.html#lineinfo,NimNode
  #
  # So `li` will look something like "/tmp/tests/foo.nim(3,0)".
  let i = li.find('(')
  if i == -1:
    return

  let path_and_filename: string = li.substr(0, i-1)
  let (_, mod_name, ext) = splitFile(path_and_filename)
  if ext != ".nim":
    return

  result = (path_and_filename, mod_name, true)


proc getModuleName*(n: NimNode): string {. compileTime .} =
  let li: string = n.lineinfo
  let (_, mod_name, _) = parseModNameFromLineinfo(li)
  return mod_name


proc verifyValidCIdent*(s: string, n: NimNode) {. compileTime .} =
  if s.len == 0:
    error("an empty string is not a valid C identifier [$1]" % lineinfo(n))

  if s[0] notin IdentStartChars:
    let msg = "string is not a valid C identifier [$1]: " % lineinfo(n)
    error(msg & s)
  for i in 1.. <s.len:
    if s[i] notin IdentChars:
      let msg = "string is not a valid C identifier [$1]: " % lineinfo(n)
      error(msg & s)


proc verifyValidPyTypeLabel*(s: string, n: NimNode) {. compileTime .} =
  if s.len == 0:
    error("an empty string is not a valid Python type label [$1]" % lineinfo(n))

  for ss in s.split('.'):
    if ss[0] notin IdentStartChars:
      let msg = "string is not a valid Python type label [$1]: " % lineinfo(n)
      error(msg & s)
    for i in 1.. <ss.len:
      if ss[i] notin IdentChars:
        let msg = "string is not a valid C identifier [$1]: " % lineinfo(n)
        error(msg & s)


proc verifyValidNimModuleName*(s: string) {. compileTime .} =
  if s.len == 0:
    error("can't import a Nim module with an empty name")

  if s[0] notin Letters:
    # NOTE:  The first character must be in Letters, not IdentStartChars
    # (in contrast to a valid C identifier).  According to the Nim manual,
    # a Nim module name must be a valid Nim identifier, and a Nim identifer
    # can't begin with an underscore (unlike a C identifier):
    #  http://nim-lang.org/manual.html#modules
    #  http://nim-lang.org/manual.html#identifiers-keywords
    let msg = "string is not a valid Nim module name: "
    error(msg & s)
  for i in 1.. <s.len:
    if s[i] notin IdentChars:
      let msg = "string is not a valid Nim module name: "
      error(msg & s)


proc verifyValidCInclude*(s: string, n: NimNode): string {. compileTime .} =
  ## Verify that the string `s` is a valid C #include path and return it;
  ## error if `s` is not a valid C #include path.
  ##
  ## Following the practice of the
  ## `header pragma <http://nim-lang.org/nimc.html#header-pragma>`_,
  ## ("As usual for C, a system header file is enclosed in angle brackets:
  ## ``<>``.  If no angle brackets are given, Nim encloses the header file
  ## in ``""`` in the generated C code."), we will enclose the string in
  ## double-quotes if there are no quotes specified.

  # For comparison, Nim's check happens in `compiler/cgen.nim` in proc
  # `generateHeaders`.
  if s.len == 0:
    error("an empty string is not a valid C #include [$1]" % lineinfo(n))
  if s.len == 1:
    let msg = "a single character ('$1') is not a valid C #include [$2]" % [s, lineinfo(n)]
    error(msg)

  let first = s[0]
  let last = s[s.high]
  if (first == '<' and last == '>') or (first == '"' and last == '"'):
    # Yes, it's valid as specified.
    return s

  if first != '<' and last != '>' and first != '"' and last != '"':
    # No quotes were provided at all.  So, following the practice of Nim's
    # header pragma, we'll enclose the string in double-quotes
    return "\"$1\"" % s

  let msg = "this string is not a valid C #include path [$1]: $2" % [lineinfo(n), s]
  error(msg)

