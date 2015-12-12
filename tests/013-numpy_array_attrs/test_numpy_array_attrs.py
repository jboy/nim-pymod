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
def random_1d_array_size(request, seeded_random_number_generator):
    """Return a random integer to be the size of an array."""
    return numpy.random.randint(1, 50)


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

    # It took me a long time to find out how to access the actual `arg.data`
    # pointer!  If you just invoke `arg.data` in Python, it returns you a
    # temporary intermediate buffer object, with a different memory address!
    data_addr = arg.__array_interface__["data"][0]
    assert res == data_addr

