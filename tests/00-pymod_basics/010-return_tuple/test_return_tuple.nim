import pymod

# TODO
#proc returnEmptyTuple*(): tuple[] {.exportpy.} = ()

proc returnOneFieldTupleNamedFields1*(x: int): tuple[a: int] {.exportpy.} = (a: x)
proc returnOneFieldTupleNamedFields2*(): tuple[a: int] {.exportpy.} = (a: 22)
proc returnOneFieldTupleNamedFields3*: tuple[a: int] {.exportpy.} = (a: 33)

proc returnTwoFieldTupleNamedFields1*(x, y: int): tuple[a: int, b: int] {.exportpy.} = (a: x, b: y)
proc returnTwoFieldTupleNamedFields2*(x, y: int): tuple[a, b: int] {.exportpy.} = (a: x, b: y)
proc returnTwoFieldTupleNamedFields3*(x: int): tuple[a, b: int] {.exportpy.} = (a: x, b: 3333)
proc returnTwoFieldTupleNamedFields4*(x: int): tuple[a, b: int] {.exportpy.} = (a: 444, b: x)
proc returnTwoFieldTupleNamedFields5*: tuple[a, b: int] {.exportpy.} = (a: 555, b: 5555)

proc returnTwoFieldTupleUnnamedFields6*(x, y: int): tuple[a, b: int] {.exportpy.} = (x, y)
proc returnTwoFieldTupleUnnamedFields7*(x: int): tuple[a, b: int] {.exportpy.} = (x, 7777)
proc returnTwoFieldTupleUnnamedFields8*(x: int): tuple[a, b: int] {.exportpy.} = (888, x)
proc returnTwoFieldTupleUnnamedFields9*: tuple[a, b: int] {.exportpy.} = (999, 9999)

initPyModule("",
    #returnEmptyTuple,
    returnOneFieldTupleNamedFields1, returnOneFieldTupleNamedFields2, returnOneFieldTupleNamedFields3,
    returnTwoFieldTupleNamedFields1, returnTwoFieldTupleNamedFields2, returnTwoFieldTupleNamedFields3,
    returnTwoFieldTupleNamedFields4, returnTwoFieldTupleNamedFields5,
    returnTwoFieldTupleUnnamedFields6, returnTwoFieldTupleUnnamedFields7,
    returnTwoFieldTupleUnnamedFields8, returnTwoFieldTupleUnnamedFields9
)
