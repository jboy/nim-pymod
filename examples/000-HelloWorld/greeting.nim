# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

## A simple "Hello, World!" example of how to use Pymod.
##
## Compile this Nim module using the following command:
##   python ../../pmgen.py greeting.nim

import strutils  # `%` operator

import pymod
import pymodpkg/docstrings


proc greet*(audience: string): string {.exportpy.} =
  docstring"""Greet the specified audience with a familiar greeting.

  The string returned will be a greeting directed specifically at that audience.
  """
  return "Hello, $1!" % audience


initPyModule("hw", greet)
