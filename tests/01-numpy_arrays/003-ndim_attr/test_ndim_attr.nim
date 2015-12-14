import pymod
import pymodpkg/pyarrayobject

proc returnNdAttr*(arr: ptr PyArrayObject): cint {.exportpy.} = arr.nd
proc returnNdimAttr*(arr: ptr PyArrayObject): cint {.exportpy.} = arr.ndim

initPyModule("", returnNdAttr, returnNdimAttr)
