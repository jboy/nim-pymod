# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

## =================
## Module docstrings
## =================
## :Author: James Boyden
## :License: MIT
## :Version: 0.1
##
## Introduction
## ------------
## This module implements Python-style docstrings that can be embedded in the
## source code of a proc, along with some simple procs and macros that enable
## those docstrings to be extracted by pragmas at compile-time.
##
## These docstrings are used for automatic extraction of introspection
## documentation from documented procs.
##
## This is an alternative to Nim's built-in double-hash documentation comments,
## since (at the time of writing) it was not possible for non-compiler code
## to extract the text from the `nnkCommentStmt
## <http://nim-lang.org/macros.html#TNimrodNodeKind>`_ nodes in the AST.
##
## The most important procs and macros provided are:
##
## 1. the `docstring` proc, which is used to declare that the string literal
##    is a docstring (using the syntax of `generalized raw string literals
##    <http://nim-lang.org/manual.html#generalized-raw-string-literals>`_)
## 2. the `extractdocstrings` macro, which is a simple example of a macro that
##    can be annotated on your docstring-containing proc `as a pragma
##    <http://nim-lang.org/manual.html#macros-as-pragmas>`_, in order to
##    extract and process docstrings
## 3. the `extractAnyDocstrings` compile-time proc, which is invoked by the
##    `extractdocstrings` pragma (or better yet, your own custom more-useful
##    pragma) to extract any docstrings in the proc
## 4. the `splitDocstringLines` compile-time proc, which you can optionally use
##    to split a sequence of long docstrings (that may well contain newlines)
##    into a sequence of single-line strings
##
## Example usage
## -------------
## This example usage code is also provided in the accompanying Nim source file
## ``testDocstrings.nim``:
##
## .. code-block:: nim
##   import docstrings
##
##   proc myProc(x: int, y: string, z: float64): int {. extractdocstrings .} =
##     ## This double-hashed comment is a Nim documentation comment:
##     ##  http://nim-lang.org/docgen.html#documentation-comments
##     ##
##     ## It's recognised by Nim's "docgen" utility, which is how the official
##     ## Nim docs are generated.  However, I can't work out how to extract the
##     ## text content of the documentation comment from my own code.
##     ##
##     ## Hence, I present extractable Python-style docstrings:
##     docstring"""This is a Python-style docstring!
##
##     It doesn't do anything by itself, but it can be extracted by pragmas like
##     the ``extractdocstrings`` pragma, enabling non-compiler/non-docgen code
##     to extract and process the text content of the docstring at compile-time.
##     """
##     echo($x, y, $z)  # just something to document...
##
##     docstring"""This is another Python-style docstring in the same proc.
##     There can be any number of docstrings amongst the top-level statements
##     of a proc.  But why would you want multiple docstrings in your proc?
##
##     Maybe you don't want a big slab of documentation right between your
##     function prototype and your function body, pushing the body way down
##     from the parameters in the prototype.  Maybe you want to document your
##     algorithm step-by-step, integrated with the code.
##     """
##
##     docstring"""The ``extractdocstrings`` pragma (or better yet, your own
##     custom pragma that invokes the ``extractAnyDocstrings`` proc, which is
##     defined in the ``docstrings`` module) will extract all docstrings.
##     """
##
##     return int(x.float64 + z)
##
##   discard myProc(1, "2", 3.0)
##
## When you compile ``testDocstrings.nim``, the docstrings will be extracted
## and printed to the console at compile-time (in the midst of the compiler
## output).  The output will look like this:
##
## ::
##   Extracted docstrings:
##   | This is a Python-style docstring!
##   |
##   | It doesn't do anything by itself, but it can be extracted by pragmas like
##   | the ``extractdocstrings`` pragma, enabling non-compiler/non-docgen code
##   | to extract and process the text content of the docstring at compile-time.
##   |
##   | This is another Python-style docstring in the same proc.
##   | There can be any number of docstrings amongst the top-level statements
##   | of a proc.  But why would you want multiple docstrings in your proc?
##   |
##   | Maybe you don't want a big slab of documentation right between your
##   | function prototype and your function body, pushing the body way down
##   | from the parameters in the prototype.  Maybe you want to document your
##   | algorithm step-by-step, integrated with the code.
##   |
##   | The ``extractdocstrings`` pragma (or better yet, your own
##   | custom pragma that invokes the ``extractAnyDocstrings`` proc, which is
##   | defined in the ``docstrings`` module) will extract all docstrings.
##   CC: testDocstrings


import macros  # expectKind, kind
import strutils  # strip


proc docstring*(s: string): void =
  ## Declare docstrings in your procs like this:
  ##
  ## .. code-block:: Nim
  ##   docstring"""This is the docstring content"""
  ##
  ## This proc doesn't actually calculate or process anything itself, but it
  ## indicates to later pragmas that the string is intended as a docstring,
  ## using `generalized raw string literals
  ## <http://nim-lang.org/manual.html#generalized-raw-string-literals>`_.
  discard


#
# Compile-time utility procs.  These aren't the procs you're looking for.
#

template lstrip(s: string): string =
  strip(s, true, false)


template rstrip(s: string): string =
  strip(s, false, true)


proc expectProcDef(proc_def_node: NimNode): string {. compileTime .} =
  expectKind(proc_def_node, nnkProcDef)
  let proc_name_node = name(proc_def_node)
  if kind(proc_name_node) == nnkEmpty:
    result = "<unnamed>"
  else:
    result = $proc_name_node


proc getDocstringStrLitNode(n: NimNode): NimNode {. compileTime .} =
  # http://nim-lang.org/manual.html#generalized-raw-string-literals
  # http://nim-lang.org/macros.html#nnkCallKinds
  if n.kind == nnkCallStrLit:
    if n.len == 2:
      let call_ident = n[0]
      if (call_ident.kind == nnkIdent and $call_ident == "docstring"):
        let arg_kind = n[1].kind
        if (arg_kind == nnkTripleStrLit or arg_kind == nnkRStrLit):
          return n[1]

  return nil


#
# Functions to extract and process docstrings.
#

proc extractAnyDocstrings*(
    proc_def_node: NimNode,
    require_docstring: bool = false): seq[string] {. compileTime .} =
  ## Extract any docstrings in the body of the supplied proc definition node.
  ##
  ## If `require_docstring` is true, a compile-time error will be triggered
  ## if no docstrings can be found in the body of the proc.
  ##
  ## A sequence of docstrings will be returned.
  result = @[]

  let proc_name = expectProcDef(proc_def_node)
  let statements = proc_def_node.body
  #hint("body = " & treeRepr(statements))
  # Since we're processing the body of the proc, we need to be ready for
  # a variety of edge-cases.  For example:
  #
  # 1. No proc body defined at all (eg, if the proc has the `importc` pragma):
  #     proc myfunc(x: int, y: float64) {. importc, extractdocstrings .}
  #    =>
  #     body = Empty
  #
  # 2. An empty proc body defined (ie, the only statement is `discard`):
  #     proc myfunc1(x: int, y: float64) {. extractdocstrings .} =
  #         discard
  #    =>
  #     body = StmtList
  #       Empty
  #       DiscardStmt
  #         Empty
  #
  # 3. An proc body defined only with a comment statement:
  #     proc myfunc1(x: int, y: float64) {. extractdocstrings .} =
  #         ## This is a comment statement.
  #    =>
  #     body = StmtList
  #       Empty
  #       CommentStmt
  #
  # More generally, if there's exactly ONE statement in the body of the proc,
  # that statement's Nimrod node will follow directly after StmtList > Empty
  # (ie, as a child of StmtList and a sibling of Empty).  But if there are
  # MULTIPLE statements in the body of the proc, they will all be contained
  # within a second INNER StmtList, that itself follows after the initial
  # StmtList > Empty.
  #
  # For example, single statement:
  #   proc myfunc1(x: int, y: float64) {. extractdocstrings .} =
  #       echo("Hello")
  #  =>
  #   body = StmtList
  #     Empty
  #     Call
  #       Sym "echo"
  #       Bracket
  #         HiddenCallConv
  #           Sym "$"
  #           StrLit Hello [User]
  #
  # versus multiple statements:
  #   proc myfunc1(x: int, y: float64) {. extractdocstrings .} =
  #       ## This is a comment statement.
  #       echo("Hello")
  #  =>
  #   body = StmtList
  #     Empty
  #     StmtList
  #       CommentStmt
  #       Call
  #         Sym "echo"
  #         Bracket
  #           HiddenCallConv
  #             Sym "$"
  #             StrLit Hello [User]
  #
  # FIXME:  This whole comment is out of date, since we changed our macro
  # parameter types from `stmt` to `expr` as per this thread on 2015-02-19:
  #  http://forum.nim-lang.org/t/894
  #
  # TODO:  Update this comment to reflect the exciting new reality.
  #
  # NOTE:  As of 2015-03-19, the structure has changed again.  Yay!
  # The `Empty` node that was the first child of the `StmtList` has gone.

  if statements == nil:
    let msg = "AST error: proc `$1` has nil body [$2]: " %
        [proc_name, proc_def_node.lineinfo]
    error(msg & treeRepr(proc_def_node))

  if statements.kind == nnkEmpty:
    if require_docstring:
      let msg = "can't extract docstrings from proc `$1` that has no body [$2]: " %
          [proc_name, proc_def_node.lineinfo]
      error(msg & treeRepr(proc_def_node))
    else:
      return

  let ds = getDocstringStrLitNode(statements)
  if ds != nil:
    # It's a single-statement body, and the statement is a docstring.
    # There is no proc body other than the docstring.
    #hint("docstring = " & ds.strVal)
    result.add(ds.strVal)
  elif statements.kind != nnkStmtList:
    # It's a single-statement body, but the statement is not
    # a string literal.
    if require_docstring:
      let msg = "no docstrings found in body of proc `$1` [$2]: " %
          [proc_name, proc_def_node.lineinfo]
      error(msg & treeRepr(statements))
    else:
      return
  else:
    # It's a multi-statement body.
    # We'll search for as many docstrings we can find at the top level
    # of statements.
    if statements.len < 1:
      # A multi-statement body that contains no statements??
      # That's not right!
      let msg = "AST error: in body of proc `$1`, expected len(StmtList) >= 1, got $2 [$3]: " %
          [proc_name, $statements.len, proc_def_node.lineinfo]
      error(msg & treeRepr(statements))
    else:  # statements.len >= 1
      let num_statements = statements.len
      for i in 0.. <num_statements:
        let ds2 = getDocstringStrLitNode(statements[i])
        if ds2 != nil:
          # Success!
          #hint("docstring = " & ds2.strVal)
          result.add(ds2.strVal)

      if result.len == 0:
        if require_docstring:
          let msg = "no docstrings found in body of proc `$1` [$2]: " %
              [proc_name, proc_def_node.lineinfo]
          error(msg & treeRepr(statements))


proc splitDocstringLines*(docstrings: seq[string]): seq[string] =
  ## Split the supplied sequence of long docstrings (which may well have been
  ## declared using triple-quoted string literals, that can contain newlines)
  ## into a sequence of single-line strings.
  ##
  ## It will also adjust the leading whitespace at the start of each line,
  ## to remove any whitespace that is attributable to the indentation of the
  ## docstring as a statement inside the proc.
  ##
  ## Usage of this proc is optional.
  result = @[]
  for ds in docstrings:
    if result.len > 0:
      # At least one docstring has already been processed.
      # Separate adjacent docstrings with an empty line.
      result.add("")

    let split_ds = ds.splitLines()
    if split_ds.len == 1:
      # Just the one line (ie, no line breaks).
      result.add(split_ds[0].rstrip())
    elif split_ds.len >= 2:
      # Multiple lines.
      # Assume the first line has no unintentional indentation.
      # BUT, it might be an EMPTY LINE (if the user starts the text
      # on the next line).  Don't bother adding an empty line.
      if split_ds[0] != "":
        result.add(split_ds[0].rstrip())

      # Now, let's remove any leading whitespace that is due to the
      # indentation of the docstring within the proc.

      let num_lines = split_ds.len
      # First, determine the minimum amount of indentation.
      # (Logic:  The amount of indentation on any given line MUST be
      # less-than-or-equal-to the total number of characters in the
      # whole of the triple-quoted string literal that is the current
      # docstring, so use that as our starting "infinity" value.)
      var minimum_indentation = ds.len + 1
      for i in 1.. <num_lines:
        let s = split_ds[i]
        let lstripped_s = s.lstrip()
        if lstripped_s.len > 0:  # ie, the line wasn't empty/whitespace
          let num_indent = s.len - lstripped_s.len
          if num_indent < minimum_indentation:
            minimum_indentation = num_indent

      # Now left-truncate every line to this minimum indentation.
      for i in 1.. <num_lines:
        let s = split_ds[i]
        result.add(s.substr(minimum_indentation).rstrip())

      # But if the last line was empty, remove it.
      # (Since any subsequent docstrings that follow, will have an
      # empty line inserted before them anyway.)
      if result[result.high] == "":
        discard result.pop()


macro extractdocstrings*(proc_def: expr): stmt =
  ## Annotate this macro as a pragma on your proc that contains docstrings,
  ## for example:
  ##
  ## .. code-block:: Nim
  ##   proc myProc(x: int, y: string, z: float64): int {. extractdocstrings .} =
  ##
  ## (If you didn't already know, `macros can be used as pragmas
  ## <http://nim-lang.org/manual.html#macros-as-pragmas>`_.)
  ##
  ## This macro is just a very simple example to demonstrate how docstrings
  ## are extracted.  You probably want to write your own macro that uses
  ## `extractAnyDocstrings` (and optionally `splitDocstringLines`) and then
  ## does something useful with the extracted text.
  let extracted_docstrings = extractAnyDocstrings(proc_def)
  if extracted_docstrings.len > 0:
    let docstring_lines = splitDocstringLines(extracted_docstrings)
    echo("Extracted docstrings:")
    for ds_line in docstring_lines:
      echo("| " & ds_line)

  # Now return the proc unchanged.
  result = proc_def

