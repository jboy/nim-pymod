import pymod

proc returnHelloWorld*(): string {.exportpy.} = "Hello World!"

initPyModule("", returnHelloWorld)
