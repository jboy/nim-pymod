import numpy
import pytest


def test_0_compile_pymod_test_mod(pmgen_py_compile):
    pmgen_py_compile(__name__)


def test_cfloatReturnArg(pymod_test_mod):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        res = pymod_test_mod.cfloatReturnArg(arg)
    assert str(excinfo.value) == "a float is required"

def test_cdoubleReturnArg(pymod_test_mod):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        res = pymod_test_mod.cdoubleReturnArg(arg)
    assert str(excinfo.value) == "a float is required"


def test_cshortReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.cshortReturnArg(arg)
    if python_major_version == 2:
        assert str(excinfo.value) == "an integer is required"
    else:  # Python 3 or above
        assert str(excinfo.value) == "an integer is required (got type NoneType)"

def test_cintReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.cintReturnArg(arg)
    if python_major_version == 2:
        assert str(excinfo.value) == "an integer is required"
    else:  # Python 3 or above
        assert str(excinfo.value) == "an integer is required (got type NoneType)"

def test_clongReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.clongReturnArg(arg)
    if python_major_version == 2:
        assert str(excinfo.value) == "an integer is required"
    else:  # Python 3 or above
        assert str(excinfo.value) == "an integer is required (got type NoneType)"


def test_cushortReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.cushortReturnArg(arg)
    if python_major_version == 2:
        assert str(excinfo.value) == "an integer is required"
    else:  # Python 3 or above
        assert str(excinfo.value) == "an integer is required (got type NoneType)"

def test_cuintReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.cuintReturnArg(arg)
    if python_major_version == 2:
        assert str(excinfo.value) == "an integer is required"
    else:  # Python 3 or above
        assert str(excinfo.value) == "an integer is required (got type NoneType)"

def test_culongReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.culongReturnArg(arg)
    assert str(excinfo.value) == "argument 1 must be integer<k>, not None"


def test_floatReturnArg(pymod_test_mod):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.floatReturnArg(arg)
    assert str(excinfo.value) == "a float is required"

def test_float32ReturnArg(pymod_test_mod):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.float32ReturnArg(arg)
    assert str(excinfo.value) == "a float is required"

def test_float64ReturnArg(pymod_test_mod):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.float64ReturnArg(arg)
    assert str(excinfo.value) == "a float is required"


def test_intReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.intReturnArg(arg)
    if python_major_version == 2:
        assert str(excinfo.value) == "an integer is required"
    else:  # Python 3 or above
        assert str(excinfo.value) == "an integer is required (got type NoneType)"

#def test_int8ReturnArg(pymod_test_mod, python_major_version):
#    arg = None
#    with pytest.raises(TypeError) as excinfo:
#        pymod_test_mod.int8ReturnArg(arg)
#    if python_major_version == 2:
#        assert str(excinfo.value) == "an integer is required"
#    else:  # Python 3 or above
#        assert str(excinfo.value) == "an integer is required (got type NoneType)"

def test_int16ReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.int16ReturnArg(arg)
    if python_major_version == 2:
        assert str(excinfo.value) == "an integer is required"
    else:  # Python 3 or above
        assert str(excinfo.value) == "an integer is required (got type NoneType)"

def test_int32ReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.int32ReturnArg(arg)
    if python_major_version == 2:
        assert str(excinfo.value) == "an integer is required"
    else:  # Python 3 or above
        assert str(excinfo.value) == "an integer is required (got type NoneType)"

def test_int64ReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.int64ReturnArg(arg)
    if python_major_version == 2:
        assert str(excinfo.value) == "an integer is required"
    else:  # Python 3 or above
        assert str(excinfo.value) == "an integer is required (got type NoneType)"


def test_uintReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.uintReturnArg(arg)
    assert str(excinfo.value) == "argument 1 must be integer<k>, not None"

def test_uint8ReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.uint8ReturnArg(arg)
    if python_major_version == 2:
        assert str(excinfo.value) == "an integer is required"
    else:  # Python 3 or above
        assert str(excinfo.value) == "an integer is required (got type NoneType)"

def test_uint16ReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.uint16ReturnArg(arg)
    if python_major_version == 2:
        assert str(excinfo.value) == "an integer is required"
    else:  # Python 3 or above
        assert str(excinfo.value) == "an integer is required (got type NoneType)"

def test_uint32ReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.uint32ReturnArg(arg)
    if python_major_version == 2:
        assert str(excinfo.value) == "an integer is required"
    else:  # Python 3 or above
        assert str(excinfo.value) == "an integer is required (got type NoneType)"

def test_uint64ReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.uint64ReturnArg(arg)
    assert str(excinfo.value) == "argument 1 must be integer<k>, not None"


#def test_boolReturnArg(pymod_test_mod):
#    arg = None
#    with pytest.raises(TypeError) as excinfo:
#        pymod_test_mod.boolReturnArg(arg)
#    assert str(excinfo.value) == "a boolean is required"

def test_byteReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.byteReturnArg(arg)
    if python_major_version == 2:
        assert str(excinfo.value) == "an integer is required"
    else:  # Python 3 or above
        assert str(excinfo.value) == "an integer is required (got type NoneType)"


def test_ccharReturnArg(pymod_test_mod, python_major_version):
    # Python 3 or above: bytes vs strings, yay!
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.ccharReturnArg(arg)
    if python_major_version == 2:
        assert str(excinfo.value) == "argument 1 must be char, not None"
    else:  # Python 3 or above
        assert str(excinfo.value) == "argument 1 must be a byte string of length 1, not None"

def test_charReturnArg(pymod_test_mod, python_major_version):
    # Python 3 or above: bytes vs strings, yay!
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.charReturnArg(arg)
    if python_major_version == 2:
        assert str(excinfo.value) == "argument 1 must be char, not None"
    else:  # Python 3 or above
        assert str(excinfo.value) == "argument 1 must be a byte string of length 1, not None"

def test_stringReturnArg(pymod_test_mod, python_major_version):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.stringReturnArg(arg)
    if python_major_version == 2:
        assert str(excinfo.value) == "argument 1 must be string, not None"
    else:  # Python 3 or above
        assert str(excinfo.value) == "argument 1 must be str, not None"


#def test_unicodeRuneReturnArg(pymod_test_mod):
#    arg = None
#    with pytest.raises(TypeError) as excinfo:
#        pymod_test_mod.unicodeRuneReturnArg(arg)
#    assert str(excinfo.value) == "argument 1 must be char, not None"

#def test_seqCharReturnArg(pymod_test_mod):
#    # Python 3 or above: bytes vs strings, yay!
#    arg = None
#    with pytest.raises(TypeError) as excinfo:
#        pymod_test_mod.seqCharReturnArg(arg)
#    assert str(excinfo.value) == "argument 1 must be string, not None"

#def test_seqRuneReturnArg(pymod_test_mod):
#    arg = None
#    with pytest.raises(TypeError) as excinfo:
#        pymod_test_mod.seqRuneReturnArg(arg)
#    assert str(excinfo.value) == "argument 1 must be string, not None"


def test_ptrPyArrayObjectReturnArg(pymod_test_mod):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.ptrPyArrayObjectReturnArg(arg)
    assert str(excinfo.value) == "argument 1 must be numpy.ndarray, not None"

def test_ptrPyObjectReturnListArg(pymod_test_mod):
    arg = None
    res = pymod_test_mod.ptrPyObjectReturnArg(arg)
    assert res == arg
    assert type(res) == type(arg)

