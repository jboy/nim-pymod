import pymod

proc returnHelloWorld*(): string {.exportpy.} = "Hello World!"

initPyModule("_pymod_test", returnHelloWorld)
