import array_utils
import numpy
import pytest


def test_0_compile_pymod_test_mod(pmgen_py_compile):
        pmgen_py_compile(__name__)


def _get_array_data_address(arr):
    # It took me a long time to find out how to access the `arr.data` address
    # (ie, obtain the actual `arr.data` pointer as an integer) in Python!
    # If you simply invoke `arr.data` in Python, it returns you a temporary
    # intermediate buffer object, that has a different memory address!
    data_addr = arr.__array_interface__["data"][0]
    return data_addr


@pytest.mark.parametrize("array_shape", array_utils.all_small_array_shapes)
@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_returnDataPointerAsInt1(pymod_test_mod, array_shape, input_type):
    arg = array_utils.get_random_Nd_array_of_shape_and_type(array_shape, input_type)
    res = pymod_test_mod.returnDataPointerAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_returnDataPointerAsInt2(pymod_test_mod, random_1d_array_size, input_type):
    arg = numpy.zeros(random_1d_array_size, dtype=input_type)
    res = pymod_test_mod.returnDataPointerAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr


def test_returnBoolDataPtrAsInt_1d1(pymod_test_mod, random_1d_array_size):
    arg = array_utils.get_random_1d_array_of_size_and_type(random_1d_array_size, numpy.bool)
    res = pymod_test_mod.returnBoolDataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnBoolDataPtrAsInt_1d2(pymod_test_mod, random_1d_array_size):
    arg = array_utils.get_random_1d_array_of_size_and_type(random_1d_array_size, numpy.bool_)
    res = pymod_test_mod.returnBoolDataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnInt8DataPtrAsInt_1d(pymod_test_mod, random_1d_array_size):
    arg = array_utils.get_random_1d_array_of_size_and_type(random_1d_array_size, numpy.int8)
    res = pymod_test_mod.returnInt8DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnInt16DataPtrAsInt_1d(pymod_test_mod, random_1d_array_size):
    arg = array_utils.get_random_1d_array_of_size_and_type(random_1d_array_size, numpy.int16)
    res = pymod_test_mod.returnInt16DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnInt32DataPtrAsInt_1d(pymod_test_mod, random_1d_array_size):
    arg = array_utils.get_random_1d_array_of_size_and_type(random_1d_array_size, numpy.int32)
    res = pymod_test_mod.returnInt32DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnInt64DataPtrAsInt_1d(pymod_test_mod, random_1d_array_size):
    arg = array_utils.get_random_1d_array_of_size_and_type(random_1d_array_size, numpy.int64)
    res = pymod_test_mod.returnInt64DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnFloat32DataPtrAsInt_1d(pymod_test_mod, random_1d_array_size):
    arg = array_utils.get_random_1d_array_of_size_and_type(random_1d_array_size, numpy.float32)
    res = pymod_test_mod.returnFloat32DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnFloat64DataPtrAsInt_1d(pymod_test_mod, random_1d_array_size):
    arg = array_utils.get_random_1d_array_of_size_and_type(random_1d_array_size, numpy.float64)
    res = pymod_test_mod.returnFloat64DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr


def test_returnBoolDataPtrAsInt_2d1(pymod_test_mod, random_2d_array_shape):
    arg = array_utils.get_random_Nd_array_of_shape_and_type(random_2d_array_shape, numpy.bool)
    res = pymod_test_mod.returnBoolDataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnBoolDataPtrAsInt_2d2(pymod_test_mod, random_2d_array_shape):
    arg = array_utils.get_random_Nd_array_of_shape_and_type(random_2d_array_shape, numpy.bool_)
    res = pymod_test_mod.returnBoolDataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnInt8DataPtrAsInt_2d(pymod_test_mod, random_2d_array_shape):
    arg = array_utils.get_random_Nd_array_of_shape_and_type(random_2d_array_shape, numpy.int8)
    res = pymod_test_mod.returnInt8DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnInt16DataPtrAsInt_2d(pymod_test_mod, random_2d_array_shape):
    arg = array_utils.get_random_Nd_array_of_shape_and_type(random_2d_array_shape, numpy.int16)
    res = pymod_test_mod.returnInt16DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnInt32DataPtrAsInt_2d(pymod_test_mod, random_2d_array_shape):
    arg = array_utils.get_random_Nd_array_of_shape_and_type(random_2d_array_shape, numpy.int32)
    res = pymod_test_mod.returnInt32DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnInt64DataPtrAsInt_2d(pymod_test_mod, random_2d_array_shape):
    arg = array_utils.get_random_Nd_array_of_shape_and_type(random_2d_array_shape, numpy.int64)
    res = pymod_test_mod.returnInt64DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnFloat32DataPtrAsInt_2d(pymod_test_mod, random_2d_array_shape):
    arg = array_utils.get_random_Nd_array_of_shape_and_type(random_2d_array_shape, numpy.float32)
    res = pymod_test_mod.returnFloat32DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnFloat64DataPtrAsInt_2d(pymod_test_mod, random_2d_array_shape):
    arg = array_utils.get_random_Nd_array_of_shape_and_type(random_2d_array_shape, numpy.float64)
    res = pymod_test_mod.returnFloat64DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr


# TODO
#def test_returnBoolDataPtrIndex0_1d(pymod_test_mod, random_1d_array_of_bool):
#    arg = random_1d_array_of_bool.copy()
#    expectedRes = bool(arg[0])
#    res = pymod_test_mod.returnBoolDataPtrIndex0(arg)
#    assert res == expectedRes
#    assert type(res) == type(expectedRes)

# TODO
#def test_returnInt8DataPtrIndex0_1d(pymod_test_mod, random_1d_array_of_integers):
#    arg = random_1d_array_of_integers.astype(numpy.int8)
#    expectedRes = int(arg[0])
#    res = pymod_test_mod.returnInt8DataPtrIndex0(arg)
#    assert res == expectedRes
#    assert type(res) == type(expectedRes)

def test_returnInt16DataPtrIndex0_1d(pymod_test_mod, random_1d_array_of_integers):
    arg = random_1d_array_of_integers.astype(numpy.int16)
    expectedRes = int(arg[0])
    res = pymod_test_mod.returnInt16DataPtrIndex0(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_returnInt32DataPtrIndex0_1d(pymod_test_mod, random_1d_array_of_integers):
    arg = random_1d_array_of_integers.astype(numpy.int32)
    expectedRes = int(arg[0])
    res = pymod_test_mod.returnInt32DataPtrIndex0(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_returnInt64DataPtrIndex0_1d(pymod_test_mod, random_1d_array_of_integers):
    arg = random_1d_array_of_integers.astype(numpy.int64)
    expectedRes = int(arg[0])
    res = pymod_test_mod.returnInt64DataPtrIndex0(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_returnFloat32DataPtrIndex0_1d(pymod_test_mod, random_1d_array_of_integers):
    arg = random_1d_array_of_integers.astype(numpy.float32)
    expectedRes = float(arg[0])
    res = pymod_test_mod.returnFloat32DataPtrIndex0(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_returnFloat64DataPtrIndex0_1d(pymod_test_mod, random_1d_array_of_integers):
    arg = random_1d_array_of_integers.astype(numpy.float64)
    expectedRes = float(arg[0])
    res = pymod_test_mod.returnFloat64DataPtrIndex0(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)


# TODO
#def test_returnBoolDataPtrIndex0_2d(pymod_test_mod, random_2d_array_of_bool):
#    arg = random_2d_array_of_bool.copy()
#    expectedRes = bool(arg[0, 0])
#    res = pymod_test_mod.returnBoolDataPtrIndex0(arg)
#    assert res == expectedRes
#    assert type(res) == type(expectedRes)

# TODO
#def test_returnInt8DataPtrIndex0_2d(pymod_test_mod, random_2d_array_of_integers):
#    arg = random_2d_array_of_integers.astype(numpy.int8)
#    expectedRes = int(arg[0, 0])
#    res = pymod_test_mod.returnInt8DataPtrIndex0(arg)
#    assert res == expectedRes
#    assert type(res) == type(expectedRes)

def test_returnInt16DataPtrIndex0_2d(pymod_test_mod, random_2d_array_of_integers):
    arg = random_2d_array_of_integers.astype(numpy.int16)
    expectedRes = int(arg[0, 0])
    res = pymod_test_mod.returnInt16DataPtrIndex0(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_returnInt32DataPtrIndex0_2d(pymod_test_mod, random_2d_array_of_integers):
    arg = random_2d_array_of_integers.astype(numpy.int32)
    expectedRes = int(arg[0, 0])
    res = pymod_test_mod.returnInt32DataPtrIndex0(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_returnInt64DataPtrIndex0_2d(pymod_test_mod, random_2d_array_of_integers):
    arg = random_2d_array_of_integers.astype(numpy.int64)
    expectedRes = int(arg[0, 0])
    res = pymod_test_mod.returnInt64DataPtrIndex0(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_returnFloat32DataPtrIndex0_2d(pymod_test_mod, random_2d_array_of_integers):
    arg = random_2d_array_of_integers.astype(numpy.float32)
    expectedRes = float(arg[0, 0])
    res = pymod_test_mod.returnFloat32DataPtrIndex0(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_returnFloat64DataPtrIndex0_2d(pymod_test_mod, random_2d_array_of_integers):
    arg = random_2d_array_of_integers.astype(numpy.float64)
    expectedRes = float(arg[0, 0])
    res = pymod_test_mod.returnFloat64DataPtrIndex0(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)

