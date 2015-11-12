# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

import macros

import pymodpkg/private/astutils
import pymodpkg/private/impls
import pymodpkg/private/registrytypes


#
#=== Registries of exported procs & defined PyObject types
#

# http://nim-lang.org/manual.html#static-statement-expression
static:
  var pyObjectTypeDefs: PyObjectTypeDefTable = @[]
  var procPrototypes: ProcPrototypeTable = @[]
  var nimModulesToImport: NimModulesToImportTable = @[]


#
#=== Register modules to be imported by the auto-generated Nim wrappers.
#

macro registerNimModuleToImport*(nimModName: string): stmt =
  nimModulesToImport.add($nimModName)


#
#=== User-invoked macros part 1: Define new PyObject types
#

macro definePyObjectType*(nimType, objCType, pyTypeObj, pyTypeLabel,
    extraIncludes: expr): stmt =
  result = definePyObjectTypeImpl(pyObjectTypeDefs,
      nimType, objCType, pyTypeObj, pyTypeLabel, extraIncludes)


#
#=== User-invoked macros part 2: exporting Nim procs to Python
#

# http://nim-lang.org/manual.html#macros-as-pragmas
macro exportpy*(procDef: expr): stmt =
  result = exportpyImpl(pyObjectTypeDefs, procPrototypes, procDef)

macro return_dict*(procDef: expr): stmt =
  result = procDef

#
#=== User-invoked macros part 3: Python C-API code generation
#

macro initPyModule*(modName: string, procNames: varargs[expr]): stmt =
  #let extraIncludes = createStrLitArray()
  #let extraInit = createStrLitArray()
  let extraIncludes = createStrLitArray("pymodpkg/private/numpyarrayobject.h")
  let extraInit = createStrLitArray("import_array();")
  result = initPyModuleImpl(
      pyObjectTypeDefs, procPrototypes, nimModulesToImport,
      modName, extraIncludes, extraInit, procNames)


# TODO:  Remove these next two macros entirely, when "pymod-extensions.cfg" is
# implemented to enable extension/customisation of Pymod extensions.

#macro initPyModuleExtra*(modName: string,
#    extraIncludes: openarray[string], extraInit: openarray[string],
#    procNames: varargs[expr]): stmt =
#  result = initPyModuleImpl(
#      pyObjectTypeDefs, procPrototypes, nimModulesToImport,
#      modName, extraIncludes, extraInit, procNames)
#
#
#macro initNumpyModule*(modName: string, procNames: varargs[expr]): stmt =
#  let extraIncludes = createStrLitArray("pymodpkg/private/numpyarrayobject.h")
#  let extraInit = createStrLitArray("import_array();")
#  result = initPyModuleImpl(
#      pyObjectTypeDefs, procPrototypes, nimModulesToImport,
#      modName, extraIncludes, extraInit, procNames)

