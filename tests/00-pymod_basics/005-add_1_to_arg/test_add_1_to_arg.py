def test_0_compile_pymod_test_mod(pmgen_py_compile):
    pmgen_py_compile(__name__)


def test_cfloatAdd1ToArg(pymod_test_mod):
    arg = 1.0
    res = pymod_test_mod.cfloatAdd1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)

def test_cdoubleAdd1ToArg(pymod_test_mod):
    arg = 1.0
    res = pymod_test_mod.cdoubleAdd1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)


def test_cshortAdd1ToArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.cshortAdd1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)

def test_cintAdd1ToArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.cintAdd1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)

def test_clongAdd1ToArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.clongAdd1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)


def test_cushortAdd1ToArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.cushortAdd1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)

def test_cuintAdd1ToArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.cuintAdd1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)

def test_culongAdd1ToArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.culongAdd1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)


def test_floatAdd1ToArg(pymod_test_mod):
    arg = 1.0
    res = pymod_test_mod.floatAdd1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)

def test_float32Add1ToArg(pymod_test_mod):
    arg = 1.0
    res = pymod_test_mod.float32Add1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)

def test_float64Add1ToArg(pymod_test_mod):
    arg = 1.0
    res = pymod_test_mod.float64Add1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)


def test_intAdd1ToArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.intAdd1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)

#def test_int8Add1ToArg(pymod_test_mod):
#    arg = 1
#    res = pymod_test_mod.int8Add1ToArg(arg)
#    assert res == (arg + 1)
#    assert type(res) == type(arg)

def test_int16Add1ToArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.int16Add1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)

def test_int32Add1ToArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.int32Add1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)

def test_int64Add1ToArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.int64Add1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)


def test_uintAdd1ToArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.uintAdd1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)

def test_uint8Add1ToArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.uint8Add1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)

def test_uint16Add1ToArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.uint16Add1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)

def test_uint32Add1ToArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.uint32Add1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)

def test_uint64Add1ToArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.uint64Add1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)


#def test_boolAdd1ToArg(pymod_test_mod):
#    arg = True
#    res = pymod_test_mod.boolAdd1ToArg(arg)
#    assert res == (arg + 1)
#    assert type(res) == type(arg)

def test_byteAdd1ToArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.byteAdd1ToArg(arg)
    assert res == (arg + 1)
    assert type(res) == type(arg)


def test_ccharAdd1ToArg(pymod_test_mod, python_major_version):
    # Python 3 or above: bytes vs strings, yay!
    arg = b"a" if python_major_version >= 3 else "a"
    expectedRes = b"b" if python_major_version >= 3 else "b"
    res = pymod_test_mod.ccharAdd1ToArg(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)
    assert type(res) == type(arg)

def test_charAdd1ToArg(pymod_test_mod, python_major_version):
    # Python 3 or above: bytes vs strings, yay!
    arg = b"a" if python_major_version >= 3 else "a"
    expectedRes = b"b" if python_major_version >= 3 else "b"
    res = pymod_test_mod.charAdd1ToArg(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)
    assert type(res) == type(arg)

def test_stringAdd1ToArg(pymod_test_mod):
    arg = "abc"
    expectedRes = arg + "def"
    res = pymod_test_mod.stringAdd1ToArg(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)
    assert type(res) == type(arg)


#def test_unicodeRuneAdd1ToArg(pymod_test_mod, python_major_version):
#    arg = "a" if python_major_version >= 3 else u"a"
#    expectedRes = "b" if python_major_version >= 3 else u"b"
#    res = pymod_test_mod.unicodeRuneAdd1ToArg(arg)
#    assert res == expectedRes
#    assert type(res) == type(expectedRes)
#    assert type(res) == type(arg)

#def test_seqCharAdd1ToArg(pymod_test_mod, python_major_version):
#    # Python 3 or above: bytes vs strings, yay!
#    arg = b"abc" if python_major_version >= 3 else "abc"
#    expectedRes = (arg + b"def") if python_major_version >= 3 else (arg + "def")
#    res = pymod_test_mod.seqCharAdd1ToArg(arg)
#    assert res == expectedRes
#    assert type(res) == type(expectedRes)
#    assert type(res) == type(arg)

#def test_seqRuneAdd1ToArg(pymod_test_mod, python_major_version):
#    arg = "abc" if python_major_version >= 3 else u"abc"
#    expectedRes = (arg + "def") if python_major_version >= 3 else (arg + u"def")
#    res = pymod_test_mod.seqRuneAdd1ToArg(arg)
#    assert res == expectedRes
#    assert type(res) == type(expectedRes)
#    assert type(res) == type(arg)

