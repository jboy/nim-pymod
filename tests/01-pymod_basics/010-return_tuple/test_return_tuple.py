import pytest
xfail = pytest.mark.xfail


def test_0_compile_pymod_test_mod(pmgen_py_compile):
    pmgen_py_compile(__name__)


#@xfail(reason="zero field tuple is incorrectly returned as None")
#def test_returnEmptyTuple(pymod_test_mod):
#    expectedRes = ()
#    res = pymod_test_mod.returnEmptyTuple()
#    assert type(res) == type(expectedRes)
#    assert res == expectedRes


@xfail(reason="one field tuple is incorrectly returned as non-tuple")
def test_returnOneFieldTupleNamedFields1(pymod_test_mod):
    arg = 1
    expectedRes = (arg,)
    res = pymod_test_mod.returnOneFieldTupleNamedFields1(arg)
    assert type(res) == type(expectedRes)
    assert res == expectedRes
    assert res[0] == expectedRes[0]

@xfail(reason="one field tuple is incorrectly returned as non-tuple")
def test_returnOneFieldTupleNamedFields2(pymod_test_mod):
    expectedRes = (22,)
    res = pymod_test_mod.returnOneFieldTupleNamedFields2()
    assert type(res) == type(expectedRes)
    assert res == expectedRes
    assert res[0] == expectedRes[0]

@xfail(reason="one field tuple is incorrectly returned as non-tuple")
def test_returnOneFieldTupleNamedFields3(pymod_test_mod):
    expectedRes = (33,)
    res = pymod_test_mod.returnOneFieldTupleNamedFields3()
    assert type(res) == type(expectedRes)
    assert res == expectedRes
    assert res[0] == expectedRes[0]


def test_returnTwoFieldTupleNamedFields1(pymod_test_mod):
    arg1 = 1
    arg2 = 11
    expectedRes = (arg1, arg2)
    res = pymod_test_mod.returnTwoFieldTupleNamedFields1(arg1, arg2)
    assert type(res) == type(expectedRes)
    assert res == expectedRes
    assert res[0] == expectedRes[0]
    assert res[1] == expectedRes[1]

def test_returnTwoFieldTupleNamedFields2(pymod_test_mod):
    arg1 = 2
    arg2 = 22
    expectedRes = (arg1, arg2)
    res = pymod_test_mod.returnTwoFieldTupleNamedFields2(arg1, arg2)
    assert type(res) == type(expectedRes)
    assert res == expectedRes
    assert res[0] == expectedRes[0]
    assert res[1] == expectedRes[1]

def test_returnTwoFieldTupleNamedFields3(pymod_test_mod):
    arg1 = 3
    expectedRes = (arg1, 3333)
    res = pymod_test_mod.returnTwoFieldTupleNamedFields3(arg1)
    assert type(res) == type(expectedRes)
    assert res == expectedRes
    assert res[0] == expectedRes[0]
    assert res[1] == expectedRes[1]

def test_returnTwoFieldTupleNamedFields4(pymod_test_mod):
    arg1 = 44
    expectedRes = (444, arg1)
    res = pymod_test_mod.returnTwoFieldTupleNamedFields4(arg1)
    assert type(res) == type(expectedRes)
    assert res == expectedRes
    assert res[0] == expectedRes[0]
    assert res[1] == expectedRes[1]

def test_returnTwoFieldTupleNamedFields5(pymod_test_mod):
    expectedRes = (555, 5555)
    res = pymod_test_mod.returnTwoFieldTupleNamedFields5()
    assert type(res) == type(expectedRes)
    assert res == expectedRes
    assert res[0] == expectedRes[0]
    assert res[1] == expectedRes[1]


def test_returnTwoFieldTupleUnnamedFields6(pymod_test_mod):
    arg1 = 6
    arg2 = 66
    expectedRes = (arg1, arg2)
    res = pymod_test_mod.returnTwoFieldTupleUnnamedFields6(arg1, arg2)
    assert type(res) == type(expectedRes)
    assert res == expectedRes
    assert res[0] == expectedRes[0]
    assert res[1] == expectedRes[1]

def test_returnTwoFieldTupleUnnamedFields7(pymod_test_mod):
    arg1 = 7
    expectedRes = (arg1, 7777)
    res = pymod_test_mod.returnTwoFieldTupleUnnamedFields7(arg1)
    assert type(res) == type(expectedRes)
    assert res == expectedRes
    assert res[0] == expectedRes[0]
    assert res[1] == expectedRes[1]

def test_returnTwoFieldTupleUnnamedFields8(pymod_test_mod):
    arg1 = 88
    expectedRes = (888, arg1)
    res = pymod_test_mod.returnTwoFieldTupleUnnamedFields8(arg1)
    assert type(res) == type(expectedRes)
    assert res == expectedRes
    assert res[0] == expectedRes[0]
    assert res[1] == expectedRes[1]

def test_returnTwoFieldTupleUnnamedFields9(pymod_test_mod):
    expectedRes = (999, 9999)
    res = pymod_test_mod.returnTwoFieldTupleUnnamedFields9()
    assert type(res) == type(expectedRes)
    assert res == expectedRes
    assert res[0] == expectedRes[0]
    assert res[1] == expectedRes[1]

