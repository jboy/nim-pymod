import pymod
import pymodpkg/pyarrayobject
import unicode


proc cfloatAdd1ToArg*(arg: cfloat):   cfloat {.exportpy.} = arg + 1.0  # No `succ`
proc cdoubleAdd1ToArg*(arg: cdouble): cdouble {.exportpy.} = arg + 1.0  # No `succ`

proc cshortAdd1ToArg*(arg: cshort):   cshort {.exportpy.} = succ(arg)
proc cintAdd1ToArg*(arg: cint):       cint {.exportpy.} = succ(arg)
proc clongAdd1ToArg*(arg: clong):     clong {.exportpy.} = succ(arg)

proc cushortAdd1ToArg*(arg: cushort): cushort {.exportpy.} = succ(arg)
proc cuintAdd1ToArg*(arg: cuint):     cuint {.exportpy.} = succ(arg)
proc culongAdd1ToArg*(arg: culong):   culong {.exportpy.} = arg + 1  # No `succ`

proc floatAdd1ToArg*(arg: float):     float {.exportpy.} = arg + 1.0  # No `succ`
proc float32Add1ToArg*(arg: float32): float32 {.exportpy.} = arg + 1.0  # No `succ`
proc float64Add1ToArg*(arg: float64): float64 {.exportpy.} = arg + 1.0  # No `succ`

proc intAdd1ToArg*(arg: int):         int {.exportpy.} = succ(arg)
#proc int8Add1ToArg*(arg: int8):       int8 {.exportpy.} = succ(arg)
proc int16Add1ToArg*(arg: int16):     int16 {.exportpy.} = succ(arg)
proc int32Add1ToArg*(arg: int32):     int32 {.exportpy.} = succ(arg)
proc int64Add1ToArg*(arg: int64):     int64 {.exportpy.} = succ(arg)

proc uintAdd1ToArg*(arg: uint):       uint {.exportpy.} = arg + 1  # No `succ`
proc uint8Add1ToArg*(arg: uint8):     uint8 {.exportpy.} = succ(arg)
proc uint16Add1ToArg*(arg: uint16):   uint16 {.exportpy.} = succ(arg)
proc uint32Add1ToArg*(arg: uint32):   uint32 {.exportpy.} = succ(arg)
proc uint64Add1ToArg*(arg: uint64):   uint64 {.exportpy.} = arg + 1  # No `succ`

#proc boolAdd1ToArg*(arg: bool):       bool {.exportpy.} = succ(arg)
proc byteAdd1ToArg*(arg: byte):       byte {.exportpy.} = succ(arg)

proc ccharAdd1ToArg*(arg: cchar):    cchar {.exportpy.} = succ(arg)
proc charAdd1ToArg*(arg: char):      char {.exportpy.} = succ(arg)
proc stringAdd1ToArg*(arg: string):  string {.exportpy.} = arg & "def"

#proc unicodeRuneAdd1ToArg*(arg: Rune):  Rune {.exportpy.} = succ(arg)
#proc seqCharAdd1ToArg*(arg: seq[char]): seq[char] {.exportpy.} = succ(arg)
#proc seqRuneAdd1ToArg*(arg: seq[Rune]): seq[Rune] {.exportpy.} = succ(arg)


initPyModule("",
    cfloatAdd1ToArg, cdoubleAdd1ToArg,
    cshortAdd1ToArg, cintAdd1ToArg, clongAdd1ToArg,
    cushortAdd1ToArg, cuintAdd1ToArg, culongAdd1ToArg,
    floatAdd1ToArg, float32Add1ToArg, float64Add1ToArg,
    #int8Add1ToArg,
    intAdd1ToArg, int16Add1ToArg, int32Add1ToArg, int64Add1ToArg,
    uintAdd1ToArg, uint8Add1ToArg, uint16Add1ToArg, uint32Add1ToArg, uint64Add1ToArg,
    #boolAdd1ToArg,
    byteAdd1ToArg,
    ccharAdd1ToArg, charAdd1ToArg, stringAdd1ToArg,
    #unicodeRuneAdd1ToArg, seqCharAdd1ToArg, seqRuneAdd1ToArg,
    )
