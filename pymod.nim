# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# [ MIT license: https://opensource.org/licenses/MIT ]


import macros

import pymodpkg/pyobject
export pyobject.PyObject


when not defined(pmgen):
  # Dummy implementations

  #
  #=== User-invoked macros part 1: Define new PyObject types
  #

  macro definePyObjectType*(nimType, objCType, pyTypeObj, pyTypeLabel,
      extraIncludes: expr): stmt =
    # Return no statements, so nothing will happen.
    result = newStmtList()


  #
  #=== User-invoked macros part 2: exporting Nim procs to Python
  #

  # http://nim-lang.org/manual.html#macros-as-pragmas
  macro exportpy*(procDef: expr): stmt =
    # The identity transformation: Return the proc unchanged.
    result = procDef


  #
  #=== User-invoked macros part 3: Python C-API code generation
  #

  macro initPyModule*(mod_name: string, proc_names: varargs[expr]): stmt =
    # Return no statements, so nothing will happen.
    result = newStmtList()


#  macro initPyModuleExtra*(modName: string,
#      extraIncludes: openArray[string], extraInit: openArray[string],
#      procNames: varargs[expr]): stmt =
#    # Return no statements, so nothing will happen.
#    result = newStmtList()
#
#
#  macro initNumpyModule*(modName: string, procNames: varargs[expr]): stmt =
#    # Return no statements, so nothing will happen.
#    result = newStmtList()


  #=== User-invoked macro: return result as a dict rather than raw tuple ===
  # Nothing actually happens in this macro;
  # the exportpy macro finds the pragma and uses it to deduce the intention
  # Will be IGNORED if included BEFORE the exportpy pragma for a given proc.
  macro return_dict*(procDef: expr): stmt =
    result = procDef
