import pymod
import unicode

template assertEqual[T](a, b: T) = doAssert(a == b)

proc cfloatExpect1*(arg: cfloat) {.exportpy.} = assertEqual(arg, 1.0)
proc cdoubleExpect1*(arg: cdouble) {.exportpy.} = assertEqual(arg, 1.0)

proc cshortExpect1*(arg: cshort) {.exportpy.} = assertEqual(arg, 1)
proc cintExpect1*(arg: cint) {.exportpy.} = assertEqual(arg, 1)
proc clongExpect1*(arg: clong) {.exportpy.} = assertEqual(arg, 1)

proc cushortExpect1*(arg: cushort) {.exportpy.} = assertEqual(arg, 1)
proc cuintExpect1*(arg: cuint) {.exportpy.} = assertEqual(arg, 1)
proc culongExpect1*(arg: culong) {.exportpy.} = assertEqual(arg, 1)

proc floatExpect1*(arg: float) {.exportpy.} = assertEqual(arg, 1.0)
proc float32Expect1*(arg: float32) {.exportpy.} = assertEqual(arg, 1.0)
proc float64Expect1*(arg: float64) {.exportpy.} = assertEqual(arg, 1.0)

proc intExpect1*(arg: int) {.exportpy.} = assertEqual(arg, 1)
# TODO
#proc int8Expect1*(arg: int8) {.exportpy.} = assertEqual(arg, 1)
proc int16Expect1*(arg: int16) {.exportpy.} = assertEqual(arg, 1)
proc int32Expect1*(arg: int32) {.exportpy.} = assertEqual(arg, 1)
proc int64Expect1*(arg: int64) {.exportpy.} = assertEqual(arg, 1)

proc uintExpect1*(arg: uint) {.exportpy.} = assertEqual(arg, 1)
proc uint8Expect1*(arg: uint8) {.exportpy.} = assertEqual(arg, 1)
proc uint16Expect1*(arg: uint16) {.exportpy.} = assertEqual(arg, 1)
proc uint32Expect1*(arg: uint32) {.exportpy.} = assertEqual(arg, 1)
proc uint64Expect1*(arg: uint64) {.exportpy.} = assertEqual(arg, 1)

# TODO
#proc boolExpect1*(arg: bool) {.exportpy.} = assertEqual(arg, True)
proc byteExpect1*(arg: byte) {.exportpy.} = assertEqual(arg, 1)

proc ccharExpect1*(arg: cchar) {.exportpy.} = assertEqual(arg, 'a')
proc charExpect1*(arg: char) {.exportpy.} = assertEqual(arg, 'a')
proc stringExpect1*(arg: string) {.exportpy.} = assertEqual(arg, "abc")

# TODO
#proc unicodeRuneExpect1*(arg: Rune) {.exportpy.} = assertEqual(arg, Rune(ord('a')))
#proc seqCharExpect1*(arg: seq[char]) {.exportpy.} = assertEqual(arg, @['a', 'b', 'c'])
#proc seqRuneExpect1*(arg: seq[Rune]) {.exportpy.} =
#  assertEqual(arg, @[Rune(ord('a')), Rune(ord('b')), Rune(ord('c'))])


initPyModule("",
    cfloatExpect1, cdoubleExpect1,
    cshortExpect1, cintExpect1, clongExpect1,
    cushortExpect1, cuintExpect1, culongExpect1,
    floatExpect1, float32Expect1, float64Expect1,
    #int8Expect1,
    intExpect1, int16Expect1, int32Expect1, int64Expect1,
    uintExpect1, uint8Expect1, uint16Expect1, uint32Expect1, uint64Expect1,
    #boolExpect1,
    byteExpect1,
    ccharExpect1, charExpect1, stringExpect1,
    #unicodeRuneExpect1, seqCharExpect1, seqRuneExpect1
    )
