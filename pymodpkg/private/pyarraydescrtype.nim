{. compile: "pymodpkg/private/pyarraydescrtype_c.c" .}

# Copyright (c) 2015 SnapDisco Pty Ltd, Australia.
# All rights reserved.
# 
# This source code is licensed under the terms of the MIT license
# found in the "LICENSE" file in the root directory of this source tree.

## Nim type definition and accessors for the Numpy C-API PyArray_Descr type.
##
## Documentation here:
##  http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArrayDescr_Type
##  http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArray_Descr


# http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArray_Descr
type PyArrayDescr* {. importc: "PyArray_Descr", header: "pymodpkg/private/numpyarrayobject.h", final .} = object


proc dkind*(d: ptr PyArrayDescr): cchar {.
    importc: "PyArrayDescr_kind", header: "pymodpkg/private/pyarraydescrtype_c.h", cdecl .}
  ## A character code indicating the kind of array (using the array interface
  ## typestring notation).
  ##
  ## A 'b' represents Boolean, a 'i' represents signed integer, a 'u' represents
  ## unsigned integer, 'f' represents floating point, 'c' represents complex
  ## floating point, 'S' represents 8-bit character string, 'U' represents
  ## 32-bit/character unicode string, and 'V' repesents arbitrary.
  ##
  ## http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArray_Descr.kind
  ##
  ## NOTE:  This attribute in the PyArrayDescr struct is called `kind`,
  ## but there's already a built-in Nim proc called `kind`.


proc dtype*(d: ptr PyArrayDescr): cchar {.
    importc: "PyArrayDescr_type", header: "pymodpkg/private/pyarraydescrtype_c.h", cdecl .}
  ## A traditional character code indicating the data type.
  ##
  ## http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArray_Descr.type
  ##
  ## NOTE:  This attribute in the PyArrayDescr struct is called `type`,
  ## but there's already a built-in Nim proc called `type`.


proc byteorder*(d: ptr PyArrayDescr): cchar {.
    importc: "PyArrayDescr_byteorder", header: "pymodpkg/private/pyarraydescrtype_c.h", cdecl .}
  ## A character indicating the byte-order:
  ## '>' (big-endian), '<' (little- endian), '=' (native),
  ## '|' (irrelevant, ignore).  All builtin data- types have byteorder '='.
  ##
  ## http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArray_Descr.byteorder


proc dflags*(d: ptr PyArrayDescr): cchar {.
    importc: "PyArrayDescr_flags", header: "pymodpkg/private/pyarraydescrtype_c.h", cdecl .}
  ## A data-type bit-flag that determines if the data-type exhibits
  ## object-array like behavior.
  ##
  ## http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArray_Descr.flags
  ##
  ## NOTE:  This attribute in the PyArrayDescr struct is called `flags`,
  ## but we want to use `flags` for a version of this proc that returns
  ## the flags parsed into a Nim set, rather than a cchar.


proc type_num*(d: ptr PyArrayDescr): cint {.
    importc: "PyArrayDescr_type_num", header: "pymodpkg/private/pyarraydescrtype_c.h", cdecl .}
  ## A number that uniquely identifies the data type.
  ##
  ## For new data-types, this number is assigned when the data-type is
  ## registered.
  ##
  ## http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArray_Descr.type_num


proc elsize*(d: ptr PyArrayDescr): cint {.
    importc: "PyArrayDescr_elsize", header: "pymodpkg/private/pyarraydescrtype_c.h", cdecl .}
  ## For data types that are always the same size (such as long), this holds
  ## the size of the data type.
  ##
  ## For flexible data types where different arrays can have a different
  ## elementsize, this should be 0.
  ##
  ## http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArray_Descr.elsize


proc alignment*(d: ptr PyArrayDescr): cint {.
    importc: "PyArrayDescr_alignment", header: "pymodpkg/private/pyarraydescrtype_c.h", cdecl .}
  ## A number providing alignment information for this data type.
  ##
  ## http://docs.scipy.org/doc/numpy/reference/c-api.types-and-structures.html#c.PyArray_Descr.alignment

