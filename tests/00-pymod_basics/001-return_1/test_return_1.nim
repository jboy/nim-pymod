import pymod
import unicode

proc cfloatReturn1*():  cfloat {.exportpy.} = 1.0
proc cdoubleReturn1*(): cdouble {.exportpy.} = 1.0

proc cshortReturn1*():  cshort {.exportpy.} = 1
proc cintReturn1*():    cint {.exportpy.} = 1
proc clongReturn1*():   clong {.exportpy.} = 1

proc cushortReturn1*(): cushort {.exportpy.} = 1
proc cuintReturn1*():   cuint {.exportpy.} = 1
proc culongReturn1*():  culong {.exportpy.} = 1

proc floatReturn1*():   float {.exportpy.} = 1.0
proc float32Return1*(): float32 {.exportpy.} = 1.0
proc float64Return1*(): float64 {.exportpy.} = 1.0

proc intReturn1*():     int {.exportpy.} = 1
# TODO
#proc int8Return1*():    int8 {.exportpy.} = 1
proc int16Return1*():   int16 {.exportpy.} = 1
proc int32Return1*():   int32 {.exportpy.} = 1
proc int64Return1*():   int64 {.exportpy.} = 1

proc uintReturn1*():    uint {.exportpy.} = 1
proc uint8Return1*():   uint8 {.exportpy.} = 1
proc uint16Return1*():  uint16 {.exportpy.} = 1
proc uint32Return1*():  uint32 {.exportpy.} = 1
proc uint64Return1*():  uint64 {.exportpy.} = 1

# TODO
#proc boolReturn1*():    bool {.exportpy.} = True
proc byteReturn1*():    byte {.exportpy.} = 1

proc ccharReturn1*():   cchar {.exportpy.} = 'a'
proc charReturn1*():    char {.exportpy.} = 'a'
proc stringReturn1*():  string {.exportpy.} = "abc"

# TODO
#proc unicodeRuneReturn1*(): Rune {.exportpy.} = Rune(ord('a'))
#proc seqCharReturn1*():     seq[char] {.exportpy.} = @['a', 'b', 'c']
#proc seqRuneReturn1*():     seq[Rune] {.exportpy.} =
#  @[Rune(ord('a')), Rune(ord('b')), Rune(ord('c'))]

proc intReturn1NoParensInDecl*: int {.exportpy.} = 1

proc noReturn* {.exportpy.} = discard
# TODO
#proc voidReturn*: void {.exportpy.} = discard


initPyModule("",
    cfloatReturn1, cdoubleReturn1,
    cshortReturn1, cintReturn1, clongReturn1,
    cushortReturn1, cuintReturn1, culongReturn1,
    floatReturn1, float32Return1, float64Return1,
    #int8Return1,
    intReturn1, int16Return1, int32Return1, int64Return1,
    uintReturn1, uint8Return1, uint16Return1, uint32Return1, uint64Return1,
    #boolReturn1,
    byteReturn1,
    ccharReturn1, charReturn1, stringReturn1,
    #unicodeRuneReturn1, seqCharReturn1, seqRuneReturn1,
    intReturn1NoParensInDecl,
    noReturn)
    #voidReturn)
