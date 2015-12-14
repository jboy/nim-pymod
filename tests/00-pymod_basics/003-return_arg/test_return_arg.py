def test_0_compile_pymod_test_mod(pmgen_py_compile):
    pmgen_py_compile(__name__)


def test_cfloatReturnArg(pymod_test_mod):
    arg = 1.0
    res = pymod_test_mod.cfloatReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

def test_cdoubleReturnArg(pymod_test_mod):
    arg = 1.0
    res = pymod_test_mod.cdoubleReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)


def test_cshortReturnArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.cshortReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

def test_cintReturnArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.cintReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

def test_clongReturnArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.clongReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)


def test_cushortReturnArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.cushortReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

def test_cuintReturnArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.cuintReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

def test_culongReturnArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.culongReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)


def test_floatReturnArg(pymod_test_mod):
    arg = 1.0
    res = pymod_test_mod.floatReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

def test_float32ReturnArg(pymod_test_mod):
    arg = 1.0
    res = pymod_test_mod.float32ReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

def test_float64ReturnArg(pymod_test_mod):
    arg = 1.0
    res = pymod_test_mod.float64ReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)


def test_intReturnArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.intReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

# TODO
#def test_int8ReturnArg(pymod_test_mod):
#    arg = 1
#    res = pymod_test_mod.int8ReturnArg(arg)
#    assert res == arg
#    assert type(res) == type(arg)

def test_int16ReturnArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.int16ReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

def test_int32ReturnArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.int32ReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

def test_int64ReturnArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.int64ReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)


def test_uintReturnArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.uintReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

def test_uint8ReturnArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.uint8ReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

def test_uint16ReturnArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.uint16ReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

def test_uint32ReturnArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.uint32ReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

def test_uint64ReturnArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.uint64ReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)


# TODO
#def test_boolReturnArg(pymod_test_mod):
#    arg = True
#    res = pymod_test_mod.boolReturnArg(arg)
#    assert res == arg
#    assert type(res) == type(arg)

def test_byteReturnArg(pymod_test_mod):
    arg = 1
    res = pymod_test_mod.byteReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)


def test_ccharReturnArg(pymod_test_mod, python_major_version):
    # Python 3 or above: bytes vs strings, yay!
    arg = b"a" if python_major_version >= 3 else "a"
    res = pymod_test_mod.ccharReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

def test_charReturnArg(pymod_test_mod, python_major_version):
    # Python 3 or above: bytes vs strings, yay!
    arg = b"a" if python_major_version >= 3 else "a"
    res = pymod_test_mod.charReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

def test_stringReturnArg(pymod_test_mod):
    arg = "abc"
    res = pymod_test_mod.stringReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)


# TODO
#def test_unicodeRuneReturnArg(pymod_test_mod, python_major_version):
#    arg = "a" if python_major_version >= 3 else u"a"
#    res = pymod_test_mod.unicodeRuneReturnArg(arg)
#    assert res == arg
#    assert type(res) == type(arg)

# TODO
#def test_seqCharReturnArg(pymod_test_mod, python_major_version):
#    # Python 3 or above: bytes vs strings, yay!
#    arg = b"abc" if python_major_version >= 3 else "abc"
#    res = pymod_test_mod.seqCharReturnArg(arg)
#    assert res == arg
#    assert type(res) == type(arg)

# TODO
#def test_seqRuneReturnArg(pymod_test_mod, python_major_version):
#    arg = "abc" if python_major_version >= 3 else u"abc"
#    res = pymod_test_mod.seqRuneReturnArg(arg)
#    assert res == arg
#    assert type(res) == type(arg)


def test_ptrPyObjectReturnListArg(pymod_test_mod):
    arg = list(range(17))
    res = pymod_test_mod.ptrPyObjectReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

