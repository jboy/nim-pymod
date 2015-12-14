import pytest


def test_0_compile_pymod_test_mod(pmgen_py_compile):
    pmgen_py_compile(__name__)


def test_cfloatExpect1(pymod_test_mod):
    pymod_test_mod.cfloatExpect1(1.0)

def test_cdoubleExpect1(pymod_test_mod):
    pymod_test_mod.cdoubleExpect1(1.0)


def test_cshortExpect1(pymod_test_mod):
    pymod_test_mod.cshortExpect1(1)

def test_cintExpect1(pymod_test_mod):
    pymod_test_mod.cintExpect1(1)

def test_clongExpect1(pymod_test_mod):
    pymod_test_mod.clongExpect1(1)


def test_cushortExpect1(pymod_test_mod):
    pymod_test_mod.cushortExpect1(1)

def test_cuintExpect1(pymod_test_mod):
    pymod_test_mod.cuintExpect1(1)

def test_culongExpect1(pymod_test_mod):
    pymod_test_mod.culongExpect1(1)


def test_floatExpect1(pymod_test_mod):
    pymod_test_mod.floatExpect1(1.0)

def test_float32Expect1(pymod_test_mod):
    pymod_test_mod.float32Expect1(1.0)

def test_float64Expect1(pymod_test_mod):
    pymod_test_mod.float64Expect1(1.0)


def test_intExpect1(pymod_test_mod):
    pymod_test_mod.intExpect1(1)

# TODO
#def test_int8Expect1(pymod_test_mod):
#    pymod_test_mod.int8Expect1(1)

def test_int16Expect1(pymod_test_mod):
    pymod_test_mod.int16Expect1(1)

def test_int32Expect1(pymod_test_mod):
    pymod_test_mod.int32Expect1(1)

def test_int64Expect1(pymod_test_mod):
    pymod_test_mod.int64Expect1(1)


def test_uintExpect1(pymod_test_mod):
    pymod_test_mod.uintExpect1(1)

def test_uint8Expect1(pymod_test_mod):
    pymod_test_mod.uint8Expect1(1)

def test_uint16Expect1(pymod_test_mod):
    pymod_test_mod.uint16Expect1(1)

def test_uint32Expect1(pymod_test_mod):
    pymod_test_mod.uint32Expect1(1)

def test_uint64Expect1(pymod_test_mod):
    pymod_test_mod.uint64Expect1(1)


# TODO
#def test_boolExpect1(pymod_test_mod):
#    pymod_test_mod.boolExpect1(True

def test_byteExpect1(pymod_test_mod):
    pymod_test_mod.byteExpect1(1)


def test_ccharExpect1(pymod_test_mod, python_major_version):
    if python_major_version == 2:
        pymod_test_mod.ccharExpect1("a")
    else:  # Python 3 or above: bytes vs strings, yay!
        pymod_test_mod.ccharExpect1(b"a")

def test_charExpect1(pymod_test_mod, python_major_version):
    if python_major_version == 2:
        pymod_test_mod.charExpect1("a")
    else:  # Python 3 or above: bytes vs strings, yay!
        pymod_test_mod.charExpect1(b"a")

def test_stringExpect1(pymod_test_mod):
    pymod_test_mod.stringExpect1("abc")


# TODO
#def test_unicodeRuneExpect1(pymod_test_mod, python_major_version):
#    if python_major_version == 2:
#        pymod_test_mod.unicodeRuneExpect1(u"a")
#    else:  # Python 3 or above: bytes vs strings, yay!
#        pymod_test_mod.unicodeRuneExpect1("a")

# TODO
#def test_seqCharExpect1(pymod_test_mod, python_major_version):
#    if python_major_version == 2:
#        pymod_test_mod.seqCharExpect1("abc")
#    else:  # Python 3 or above: bytes vs strings, yay!
#        pymod_test_mod.seqCharExpect1(b"abc")

# TODO
#def test_seqRuneExpect1(pymod_test_mod, python_major_version):
#    if python_major_version == 2:
#        pymod_test_mod.seqRuneExpect1(u"abc")
#    else:  # Python 3 or above: bytes vs strings, yay!
#        pymod_test_mod.seqRuneExpect1("abc")


def test_floatExpect_but_supply_int(pymod_test_mod, python_major_version):
    pymod_test_mod.floatExpect1(1)

def test_floatExpect_but_supply_str(pymod_test_mod, python_major_version):
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.floatExpect1('a')
    if python_major_version == 2:
        assert str(excinfo.value) == "a float is required"
    else:  # Python 3 or above
        assert str(excinfo.value) == "a float is required"


def test_intExpect_but_supply_float(pymod_test_mod, python_major_version):
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.intExpect1(1.0)
    if python_major_version == 2:
        assert str(excinfo.value) == "integer argument expected, got float"
    else:  # Python 3 or above
        assert str(excinfo.value) == "integer argument expected, got float"

def test_intExpect_but_supply_str(pymod_test_mod, python_major_version):
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.intExpect1('a')
    if python_major_version == 2:
        assert str(excinfo.value) == "an integer is required"
    else:  # Python 3 or above
        assert str(excinfo.value) == "an integer is required (got type str)"


def test_stringExpect_but_supply_float(pymod_test_mod, python_major_version):
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.stringExpect1(1.0)
    if python_major_version == 2:
        assert str(excinfo.value) == "argument 1 must be string, not float"
    else:  # Python 3 or above
        assert str(excinfo.value) == "argument 1 must be str, not float"

def test_stringExpect_but_supply_int(pymod_test_mod, python_major_version):
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.stringExpect1(1)
    if python_major_version == 2:
        assert str(excinfo.value) == "argument 1 must be string, not int"
    else:  # Python 3 or above
        assert str(excinfo.value) == "argument 1 must be str, not int"


def test_charExpect_but_supply_str(pymod_test_mod, python_major_version):
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.charExpect1("abc")
    if python_major_version == 2:
        assert str(excinfo.value) == "argument 1 must be char, not str"
    else:  # Python 3 or above
        assert str(excinfo.value) == "argument 1 must be a byte string of length 1, not str"

