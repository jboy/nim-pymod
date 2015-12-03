import pymod
import unicode

proc cfloatReturnArg*(arg: cfloat):   cfloat {.exportpy.} = arg
proc cdoubleReturnArg*(arg: cdouble): cdouble {.exportpy.} = arg

proc cshortReturnArg*(arg: cshort):   cshort {.exportpy.} = arg
proc cintReturnArg*(arg: cint):       cint {.exportpy.} = arg
proc clongReturnArg*(arg: clong):     clong {.exportpy.} = arg

proc cushortReturnArg*(arg: cushort): cushort {.exportpy.} = arg
proc cuintReturnArg*(arg: cuint):     cuint {.exportpy.} = arg
proc culongReturnArg*(arg: culong):   culong {.exportpy.} = arg

proc floatReturnArg*(arg: float):     float {.exportpy.} = arg
proc float32ReturnArg*(arg: float32): float32 {.exportpy.} = arg
proc float64ReturnArg*(arg: float64): float64 {.exportpy.} = arg

proc intReturnArg*(arg: int):         int {.exportpy.} = arg
#proc int8ReturnArg*(arg: int8):       int8 {.exportpy.} = arg
proc int16ReturnArg*(arg: int16):     int16 {.exportpy.} = arg
proc int32ReturnArg*(arg: int32):     int32 {.exportpy.} = arg
proc int64ReturnArg*(arg: int64):     int64 {.exportpy.} = arg

proc uintReturnArg*(arg: uint):       uint {.exportpy.} = arg
proc uint8ReturnArg*(arg: uint8):     uint8 {.exportpy.} = arg
proc uint16ReturnArg*(arg: uint16):   uint16 {.exportpy.} = arg
proc uint32ReturnArg*(arg: uint32):   uint32 {.exportpy.} = arg
proc uint64ReturnArg*(arg: uint64):   uint64 {.exportpy.} = arg

#proc boolReturnArg*(arg: bool):       bool {.exportpy.} = arg
proc byteReturnArg*(arg: byte):       byte {.exportpy.} = arg

proc ccharReturnArg*(arg: cchar):    cchar {.exportpy.} = arg
proc charReturnArg*(arg: char):      char {.exportpy.} = arg
proc stringReturnArg*(arg: string):  string {.exportpy.} = arg

#proc unicodeRuneReturnArg*(arg: Rune):  Rune {.exportpy.} = arg
#proc seqCharReturnArg*(arg: seq[char]): seq[char] {.exportpy.} = arg
#proc seqRuneReturnArg*(arg: seq[Rune]): seq[Rune] {.exportpy.} = arg


initPyModule("",
    cfloatReturnArg, cdoubleReturnArg,
    cshortReturnArg, cintReturnArg, clongReturnArg,
    cushortReturnArg, cuintReturnArg, culongReturnArg,
    floatReturnArg, float32ReturnArg, float64ReturnArg,
    #int8ReturnArg,
    intReturnArg, int16ReturnArg, int32ReturnArg, int64ReturnArg,
    uintReturnArg, uint8ReturnArg, uint16ReturnArg, uint32ReturnArg, uint64ReturnArg,
    #boolReturnArg,
    byteReturnArg,
    ccharReturnArg, charReturnArg, stringReturnArg,
    #unicodeRuneReturnArg, seqCharReturnArg, seqRuneReturnArg,
    )
