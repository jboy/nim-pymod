import pymod
import pymodpkg/pyarrayobject


proc returnPyArrayObjectPtrAsInt*(arr: ptr PyArrayObject): int {.exportpy.} = cast[int](arr)

initPyModule("", returnPyArrayObjectPtrAsInt)
