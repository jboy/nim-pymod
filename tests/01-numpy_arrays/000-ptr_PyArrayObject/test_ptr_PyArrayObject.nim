import pymod
import pymodpkg/pyarrayobject

proc returnPyArrayObjectPtrAsInt*(arr: ptr PyArrayObject): int {.exportpy.} = cast[int](arr)

proc ptrPyArrayObjectReturnArg*(arg: ptr PyArrayObject): ptr PyArrayObject {.exportpy.} = arg

initPyModule("", returnPyArrayObjectPtrAsInt, ptrPyArrayObjectReturnArg)
