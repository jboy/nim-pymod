import pymod
import pymodpkg/pyarrayobject

proc returnDtypeAsString*(arr: ptr PyArrayObject): string {.exportpy.} = $arr.dtype

initPyModule("", returnDtypeAsString)
