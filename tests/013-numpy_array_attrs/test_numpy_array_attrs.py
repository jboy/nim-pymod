import numpy
import pytest
import time


def test_0_compile_pymod_test_mod(pmgen_py_compile):
        pmgen_py_compile(__name__)


@pytest.fixture(scope="module")
def seeded_random_number_generator():
    """Ensure that the random number generator has been seeded."""
    numpy.random.seed(int(time.time()))


@pytest.fixture
def random_1d_array_size(seeded_random_number_generator):
    """Return a random integer in the range [1, 20] to be the size of a 1-D array."""
    return numpy.random.randint(20) + 1  # So the size is always >= 1.


@pytest.fixture
def random_2d_array_shape(seeded_random_number_generator):
    """Return a tuple of 2 random integers in the range [1, 10] to be the shape of a 2-D array."""
    dim1 = numpy.random.randint(10) + 1  # So the size is always >= 1.
    dim2 = numpy.random.randint(10) + 1  # So the size is always >= 1.
    return (dim1, dim2)


@pytest.fixture
def random_1d_array_of_bool(random_1d_array_size):
    """Return a randomly-sized array of random bool values."""
    return numpy.random.random_integers(0, 1, random_1d_array_size).astype(numpy.bool)


@pytest.fixture
def random_1d_array_of_integers(random_1d_array_size):
    """Return a randomly-sized 1-D array of random integers in the range [-100, 100]."""
    return numpy.random.random_integers(-100, 100, random_1d_array_size)


@pytest.fixture
def random_2d_array_of_integers(random_2d_array_shape):
    """Return a randomly-shaped 2-D array of random integers in the range [-100, 100]."""
    array_size = numpy.prod(random_2d_array_shape)
    return numpy.random.random_integers(-100, 100, array_size).reshape(random_2d_array_shape)


@pytest.mark.parametrize("input_type", [
        numpy.bool,
        numpy.int8,
        numpy.int16,
        numpy.int32,
        numpy.int64,
        numpy.uint8,
        numpy.uint16,
        numpy.uint32,
        numpy.uint64,
        numpy.float32,
        numpy.float64,
])
def test_returnPyArrayObjectPtrAsInt(pymod_test_mod, random_1d_array_size, input_type):
    arg = numpy.zeros(random_1d_array_size, dtype=input_type)
    res = pymod_test_mod.returnPyArrayObjectPtrAsInt(arg)
    assert res == id(arg)


@pytest.mark.parametrize("input_type,input_type_str", [
        # NOTE:  Pymod returns the type string "<type 'numpy.bool'>" rather than "<type 'bool'>",
        # which is what `str(numpy.bool)` returns in Python.
        # Pymod returns "<type 'numpy.bool'>" for consistency with all other Numpy type strings,
        # which are all of the form "<type 'numpy.xxxx'>" (eg, "<type 'numpy.int8'>").
        (numpy.bool,    "numpy.bool"),
        (numpy.int8,    "numpy.int8"),
        (numpy.int16,   "numpy.int16"),
        (numpy.int32,   "numpy.int32"),
        (numpy.int64,   "numpy.int64"),
        (numpy.uint8,   "numpy.uint8"),
        (numpy.uint16,  "numpy.uint16"),
        (numpy.uint32,  "numpy.uint32"),
        (numpy.uint64,  "numpy.uint64"),
        (numpy.float32, "numpy.float32"),
        (numpy.float64, "numpy.float64"),
])
def test_returnDtypeAsString(pymod_test_mod, random_1d_array_size, input_type, input_type_str):
    arg = numpy.zeros(random_1d_array_size, dtype=input_type)
    res = pymod_test_mod.returnDtypeAsString(arg)
    assert res == input_type_str


def _get_array_data_address(arr):
    # It took me a long time to find out how to access the `arr.data` address
    # (ie, obtain the actual `arr.data` pointer as an integer) in Python!
    # If you simply invoke `arr.data` in Python, it returns you a temporary
    # intermediate buffer object, that has a different memory address!
    data_addr = arr.__array_interface__["data"][0]
    return data_addr


@pytest.mark.parametrize("input_type", [
        numpy.bool,
        numpy.int8,
        numpy.int16,
        numpy.int32,
        numpy.int64,
        numpy.uint8,
        numpy.uint16,
        numpy.uint32,
        numpy.uint64,
        numpy.float32,
        numpy.float64,
])
def test_returnDataPointerAsInt(pymod_test_mod, random_1d_array_size, input_type):
    arg = numpy.zeros(random_1d_array_size, dtype=input_type)
    res = pymod_test_mod.returnDataPointerAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr


def test_returnBoolDataPtrAsInt(pymod_test_mod, random_1d_array_size):
    arg = numpy.zeros(random_1d_array_size, dtype=numpy.bool)
    res = pymod_test_mod.returnBoolDataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnInt8DataPtrAsInt(pymod_test_mod, random_1d_array_size):
    arg = numpy.zeros(random_1d_array_size, dtype=numpy.int8)
    res = pymod_test_mod.returnInt8DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnInt16DataPtrAsInt(pymod_test_mod, random_1d_array_size):
    arg = numpy.zeros(random_1d_array_size, dtype=numpy.int16)
    res = pymod_test_mod.returnInt16DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnInt32DataPtrAsInt(pymod_test_mod, random_1d_array_size):
    arg = numpy.zeros(random_1d_array_size, dtype=numpy.int32)
    res = pymod_test_mod.returnInt32DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnInt64DataPtrAsInt(pymod_test_mod, random_1d_array_size):
    arg = numpy.zeros(random_1d_array_size, dtype=numpy.int64)
    res = pymod_test_mod.returnInt64DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnFloat32DataPtrAsInt(pymod_test_mod, random_1d_array_size):
    arg = numpy.zeros(random_1d_array_size, dtype=numpy.float32)
    res = pymod_test_mod.returnFloat32DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr

def test_returnFloat64DataPtrAsInt(pymod_test_mod, random_1d_array_size):
    arg = numpy.zeros(random_1d_array_size, dtype=numpy.float64)
    res = pymod_test_mod.returnFloat64DataPtrAsInt(arg)
    data_addr = _get_array_data_address(arg)
    assert res == data_addr


#def test_returnBoolDataPtrIndex0(pymod_test_mod, random_1d_array_of_bool):
#    arg = random_1d_array_of_bool.copy()
#    expectedRes = bool(arg[0])
#    res = pymod_test_mod.returnBoolDataPtrIndex0(arg)
#    assert res == expectedRes
#    assert type(res) == type(expectedRes)

#def test_returnInt8DataPtrIndex0(pymod_test_mod, random_1d_array_of_integers):
#    arg = random_1d_array_of_integers.astype(numpy.int8)
#    expectedRes = int(arg[0])
#    res = pymod_test_mod.returnInt8DataPtrIndex0(arg)
#    assert res == expectedRes
#    assert type(res) == type(expectedRes)

def test_returnInt16DataPtrIndex0(pymod_test_mod, random_1d_array_of_integers):
    arg = random_1d_array_of_integers.astype(numpy.int16)
    expectedRes = int(arg[0])
    res = pymod_test_mod.returnInt16DataPtrIndex0(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_returnInt32DataPtrIndex0(pymod_test_mod, random_1d_array_of_integers):
    arg = random_1d_array_of_integers.astype(numpy.int32)
    expectedRes = int(arg[0])
    res = pymod_test_mod.returnInt32DataPtrIndex0(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_returnInt64DataPtrIndex0(pymod_test_mod, random_1d_array_of_integers):
    arg = random_1d_array_of_integers.astype(numpy.int64)
    expectedRes = int(arg[0])
    res = pymod_test_mod.returnInt64DataPtrIndex0(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_returnFloat32DataPtrIndex0(pymod_test_mod, random_1d_array_of_integers):
    arg = random_1d_array_of_integers.astype(numpy.float32)
    expectedRes = float(arg[0])
    res = pymod_test_mod.returnFloat32DataPtrIndex0(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)

def test_returnFloat64DataPtrIndex0(pymod_test_mod, random_1d_array_of_integers):
    arg = random_1d_array_of_integers.astype(numpy.float64)
    expectedRes = float(arg[0])
    res = pymod_test_mod.returnFloat64DataPtrIndex0(arg)
    assert res == expectedRes
    assert type(res) == type(expectedRes)


@pytest.mark.parametrize("input_type", [
        numpy.bool,
        numpy.int8,
        numpy.int16,
        numpy.int32,
        numpy.int64,
        numpy.uint8,
        numpy.uint16,
        numpy.uint32,
        numpy.uint64,
        numpy.float32,
        numpy.float64,
])
def test_returnNdAttr_and_returnNdimAttr_1d(pymod_test_mod, input_type, random_1d_array_of_integers):
    arg = random_1d_array_of_integers.astype(input_type)

    resNd = pymod_test_mod.returnNdAttr(arg)
    assert resNd == len(arg.shape)
    assert resNd == arg.ndim

    resNdim = pymod_test_mod.returnNdimAttr(arg)
    assert resNdim == len(arg.shape)
    assert resNdim == arg.ndim


@pytest.mark.parametrize("input_type", [
        numpy.bool,
        numpy.int8,
        numpy.int16,
        numpy.int32,
        numpy.int64,
        numpy.uint8,
        numpy.uint16,
        numpy.uint32,
        numpy.uint64,
        numpy.float32,
        numpy.float64,
])
def test_returnNdAttr_and_returnNdimAttr_2d(pymod_test_mod, input_type, random_2d_array_of_integers):
    arg = random_2d_array_of_integers.astype(input_type)

    resNd = pymod_test_mod.returnNdAttr(arg)
    assert resNd == len(arg.shape)
    assert resNd == arg.ndim

    resNdim = pymod_test_mod.returnNdimAttr(arg)
    assert resNdim == len(arg.shape)
    assert resNdim == arg.ndim

