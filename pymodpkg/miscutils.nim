# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import math  # ceil(x: float), log10(x: float)
import strutils  # find(s: string, sub: char), toHex(x: BiggestInt; len: int), ...
import typetraits  # name(t: typedesc)


template ilog16ceil(i: int): int =
  # Requires `math` module
  ceil(log10(i.float) / log10(16.0)).int


proc toHex*(p: pointer): string {. inline .} =
  ## Get the familiar hexadecimal representation of a pointer value.
  # Requires `srtutils` module
  let ip: int = cast[int](p)
  let num_digits = ilog16ceil(ip)
  result = "0x" & ip.toHex(num_digits).toLower


proc toHex*[T](p: ptr T): string {. inline .} =
  ## Get the familiar hexadecimal representation of a pointer value.
  # Requires `srtutils` module
  let ip: int = cast[int](p)
  let num_digits = ilog16ceil(ip)
  result = "0x" & ip.toHex(num_digits).toLower


proc toStr*[R, T](arr: array[R, T]): string =
  ## Format array `arr` as a string, because there is no `$` for arrays.
  result = ($(@arr)).substr(1)


proc getCompileTimeType*(t: typedesc): string =
  ## Given a typedesc (the symbol that identifies a type, eg. `int32`),
  ## return the name of the type.
  ## This is useful to get the name of generic parameters.
  result = name(t)


proc getGenericTypeName*[X](x: X): string =
  ## Given an argument of type `X`, assume that `X` is an
  ## instantiated generic type of the form "TypeName[Params]".
  ## Extract and return the generic type name ("TypeName").
  let n = x.type.name
  let bracket_idx = n.find('[')
  if bracket_idx == -1:
    # It appears that `X` is not actually a generic type.
    result = n
  else:
    result = n.substr(0, bracket_idx-1)  # -1 to exclude the `[`


proc parseStackTraceLine(line: string,
    filename: var string, linenum: var int, funcname: var string): bool =
  let i = line.find('(')
  if i == -1:
    return false
  let filename_part: string = line.substr(0, i-1)
  let rest: string = line.substr(i+1)

  let j = rest.find(')')
  if j == -1:
    return false
  let linenum_part = rest.substr(0, j-1)
  let funcname_part = rest.substr(j+1)
  for c in linenum_part:
    if c notin Digits:
      return false

  filename = filename_part
  linenum = parseInt(linenum_part)
  funcname = funcname_part.strip()

  return true


proc prettyPrintStackTrace*(input: string): string =
  ## Re-format the default (ugly) Nim stack track to look more like Python.
  let lines = input.splitLines()
  let num_lines = lines.len
  if num_lines == 0:
    # No stacktrace; nothing to do.
    return input
  var output_lines: seq[string] = @[]

  # Parse the first line of the stack trace; it should look like this:
  #   Traceback (most recent call last)
  var lines0 = lines[0]
  if lines0.startswith("Traceback"):
    lines0 = "Nim t" & lines0.substr(1)
  if lines0.endswith(")"):
    lines0 = lines0 & ":"  # A bit more Python-like...
  output_lines.add(lines0)

  # Parse the remaining lines of the stack trace; they should look like this:
  #   testpymod3_pymod_wrap.nim(102) exportpy_myNumpyAdd
  #   testPymod.nim(163)       myNumpyAdd
  #   pyarrayiterators.nim(51) []
  for i in 1.. <num_lines:
    let line = lines[i]
    var filename, funcname: string
    var linenum: int
    if not parseStackTraceLine(line, filename, linenum, funcname):
      # Unable to parse line.  Just store it, unparsed.
      output_lines.add(line)
      continue

    # This is also more Python-like...
    let s = "  File \"$1\", line $2, in $3" % [filename, $linenum, funcname]
    output_lines.add(s)

  result = output_lines.join("\n")

