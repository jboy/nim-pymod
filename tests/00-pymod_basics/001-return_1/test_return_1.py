def test_0_compile_pymod_test_mod(pmgen_py_compile):
    pmgen_py_compile(__name__)


def test_cfloatReturn1(pymod_test_mod):
    expectedRes = 1.0
    res = pymod_test_mod.cfloatReturn1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_cdoubleReturn1(pymod_test_mod):
    expectedRes = 1.0
    res = pymod_test_mod.cdoubleReturn1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)


def test_cshortReturn1(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.cshortReturn1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_cintReturn1(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.cintReturn1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_clongReturn1(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.clongReturn1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)


def test_cushortReturn1(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.cushortReturn1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_cuintReturn1(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.cuintReturn1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_culongReturn1(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.culongReturn1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)


def test_floatReturn1(pymod_test_mod):
    expectedRes = 1.0
    res = pymod_test_mod.floatReturn1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_float32Return1(pymod_test_mod):
    expectedRes = 1.0
    res = pymod_test_mod.float32Return1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_float64Return1(pymod_test_mod):
    expectedRes = 1.0
    res = pymod_test_mod.float64Return1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)


def test_intReturn1(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.intReturn1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

# TODO
#def test_int8Return1(pymod_test_mod):
#    expectedRes = 1
#    res = pymod_test_mod.int8Return1()
#    assert res == expectedRes
#    assert type(res) == type(expectedRes)

def test_int16Return1(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.int16Return1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_int32Return1(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.int32Return1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_int64Return1(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.int64Return1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)


def test_uintReturn1(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.uintReturn1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_uint8Return1(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.uint8Return1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_uint16Return1(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.uint16Return1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_uint32Return1(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.uint32Return1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_uint64Return1(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.uint64Return1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)


# TODO
#def test_boolReturn1(pymod_test_mod):
#    expectedRes = True
#    res = pymod_test_mod.boolReturn1()
#    assert res == expectedRes
#    assert type(res) == type(expectedRes)

def test_byteReturn1(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.byteReturn1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)


def test_ccharReturn1(pymod_test_mod, python_major_version):
    # Python 3 or above: bytes vs strings, yay!
    expectedRes = b"a" if python_major_version >= 3 else "a"
    res = pymod_test_mod.ccharReturn1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_charReturn1(pymod_test_mod, python_major_version):
    # Python 3 or above: bytes vs strings, yay!
    expectedRes = b"a" if python_major_version >= 3 else "a"
    res = pymod_test_mod.charReturn1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_stringReturn1(pymod_test_mod):
    expectedRes = "abc"
    res = pymod_test_mod.stringReturn1()
    assert res == expectedRes
    assert type(res) == type(expectedRes)


# TODO
#def test_unicodeRuneReturn1(pymod_test_mod, python_major_version):
#    expectedRes = "a" if python_major_version >= 3 else u"a"
#    res = pymod_test_mod.unicodeRuneReturn1()
#    assert res == expectedRes
#    assert type(res) == type(expectedRes)

# TODO
#def test_seqCharReturn1(pymod_test_mod, python_major_version):
#    # Python 3 or above: bytes vs strings, yay!
#    expectedRes = b"abc" if python_major_version >= 3 else "abc"
#    res = pymod_test_mod.seqCharReturn1()
#    assert res == expectedRes
#    assert type(res) == type(expectedRes)

# TODO
#def test_seqRuneReturn1(pymod_test_mod, python_major_version):
#    expectedRes = "abc" if python_major_version >= 3 else u"abc"
#    res = pymod_test_mod.seqRuneReturn1()
#    assert res == expectedRes
#    assert type(res) == type(expectedRes)


def test_intReturn1NoParensInDecl(pymod_test_mod):
    expectedRes = 1
    res = pymod_test_mod.intReturn1NoParensInDecl()
    assert res == expectedRes
    assert type(res) == type(expectedRes)


def test_noReturn(pymod_test_mod):
    expectedRes = None
    res = pymod_test_mod.noReturn()
    assert res == expectedRes
    assert type(res) == type(expectedRes)

# TODO
#def test_voidReturn(pymod_test_mod):
#    expectedRes = None
#    res = pymod_test_mod.voidReturn()
#    assert res == expectedRes
#    assert type(res) == type(expectedRes)

