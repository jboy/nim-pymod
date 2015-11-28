import pymod

proc cfloatReturn1*():  cfloat {.exportpy.} = 1.0
proc cdoubleReturn1*(): cdouble {.exportpy.} = 1.0

#proc cscharReturn1*():  cschar {.exportpy.} = 1
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
#proc int8Return1*():    int8 {.exportpy.} = 1
proc int16Return1*():   int16 {.exportpy.} = 1
proc int32Return1*():   int32 {.exportpy.} = 1
proc int64Return1*():   int64 {.exportpy.} = 1

proc uintReturn1*():    uint {.exportpy.} = 1
proc uint8Return1*():   uint8 {.exportpy.} = 1
proc uint16Return1*():  uint16 {.exportpy.} = 1
proc uint32Return1*():  uint32 {.exportpy.} = 1
proc uint64Return1*():  uint64 {.exportpy.} = 1

proc byteReturn1*():    byte {.exportpy.} = 1

proc ccharReturn1*():   cchar {.exportpy.} = '1'
proc charReturn1*():    char {.exportpy.} = '1'
proc cucharReturn1*():  cuchar {.exportpy.} = '1'
proc stringReturn1*():  string {.exportpy.} = "one"


initPyModule("_pymod_test",
    cfloatReturn1, cdoubleReturn1,
    #cscharReturn1,
    cshortReturn1, cintReturn1, clongReturn1,
    cushortReturn1, cuintReturn1, culongReturn1,
    floatReturn1, float32Return1, float64Return1,
    #int8Return1,
    intReturn1, int16Return1, int32Return1, int64Return1,
    uintReturn1, uint8Return1, uint16Return1, uint32Return1, uint64Return1,
    byteReturn1,
    ccharReturn1, charReturn1, cucharReturn1, stringReturn1)
