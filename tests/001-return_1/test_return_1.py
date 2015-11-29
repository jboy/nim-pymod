def test_0_compile_pymod_test_mod(pmgen_py_compile):
    pmgen_py_compile(__name__)


def test_cfloatReturn1(pymod_test_mod):
    print(dir(pymod_test_mod))
    assert pymod_test_mod.cfloatReturn1() == 1.0

def test_cdoubleReturn1(pymod_test_mod):
    assert pymod_test_mod.cdoubleReturn1() == 1.0


def test_cshortReturn1(pymod_test_mod):
    assert pymod_test_mod.cshortReturn1() == 1

def test_cintReturn1(pymod_test_mod):
    assert pymod_test_mod.cintReturn1() == 1

def test_clongReturn1(pymod_test_mod):
    assert pymod_test_mod.clongReturn1() == 1


def test_cushortReturn1(pymod_test_mod):
    assert pymod_test_mod.cushortReturn1() == 1

def test_cuintReturn1(pymod_test_mod):
    assert pymod_test_mod.cuintReturn1() == 1

def test_culongReturn1(pymod_test_mod):
    assert pymod_test_mod.culongReturn1() == 1


def test_floatReturn1(pymod_test_mod):
    assert pymod_test_mod.floatReturn1() == 1.0

def test_float32Return1(pymod_test_mod):
    assert pymod_test_mod.float32Return1() == 1.0

def test_float64Return1(pymod_test_mod):
    assert pymod_test_mod.float64Return1() == 1.0


def test_intReturn1(pymod_test_mod):
    assert pymod_test_mod.intReturn1() == 1

#def test_int8Return1(pymod_test_mod):
#    assert pymod_test_mod.int8Return1() == 1

def test_int16Return1(pymod_test_mod):
    assert pymod_test_mod.int16Return1() == 1

def test_int32Return1(pymod_test_mod):
    assert pymod_test_mod.int32Return1() == 1

def test_int64Return1(pymod_test_mod):
    assert pymod_test_mod.int64Return1() == 1


def test_uintReturn1(pymod_test_mod):
    assert pymod_test_mod.uintReturn1() == 1

def test_uint8Return1(pymod_test_mod):
    assert pymod_test_mod.uint8Return1() == 1

def test_uint16Return1(pymod_test_mod):
    assert pymod_test_mod.uint16Return1() == 1

def test_uint32Return1(pymod_test_mod):
    assert pymod_test_mod.uint32Return1() == 1

def test_uint64Return1(pymod_test_mod):
    assert pymod_test_mod.uint64Return1() == 1


#def test_boolReturn1(pymod_test_mod):
#    assert pymod_test_mod.boolReturn1() == True

def test_byteReturn1(pymod_test_mod):
    assert pymod_test_mod.byteReturn1() == 1


#def test_unicodeRuneReturn1(pymod_test_mod, python_major_version):
#    if python_major_version == 2:
#        assert pymod_test_mod.unicodeRuneReturn1() == u"a"
#    else:  # Python 3 or above: bytes vs strings, yay!
#        assert pymod_test_mod.unicodeRuneReturn1() == "a"

def test_ccharReturn1(pymod_test_mod, python_major_version):
    if python_major_version == 2:
        assert pymod_test_mod.ccharReturn1() == "a"
    else:  # Python 3 or above: bytes vs strings, yay!
        assert pymod_test_mod.ccharReturn1() == b"a"

def test_charReturn1(pymod_test_mod, python_major_version):
    if python_major_version == 2:
        assert pymod_test_mod.charReturn1() == "a"
    else:  # Python 3 or above: bytes vs strings, yay!
        assert pymod_test_mod.charReturn1() == b"a"

def test_stringReturn1(pymod_test_mod):
    assert pymod_test_mod.stringReturn1() == "one"


def test_intReturn1NoParensInDecl(pymod_test_mod):
    assert pymod_test_mod.intReturn1NoParensInDecl() == 1

