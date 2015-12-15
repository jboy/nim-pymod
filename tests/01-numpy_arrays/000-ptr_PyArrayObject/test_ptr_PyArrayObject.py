import array_utils
import numpy
import pytest


def test_0_compile_pymod_test_mod(pmgen_py_compile):
        pmgen_py_compile(__name__)


@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_returnPyArrayObjectPtrAsInt(pymod_test_mod, random_1d_array_size, input_type):
    arg = array_utils.get_random_1d_array_of_size_and_type(random_1d_array_size, input_type)
    res = pymod_test_mod.returnPyArrayObjectPtrAsInt(arg)
    assert res == id(arg)


@pytest.mark.parametrize("array_shape", array_utils.all_small_array_shapes)
@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_ptrPyArrayObjectReturnArg1(pymod_test_mod, array_shape, input_type):
    arg = array_utils.get_random_Nd_array_of_shape_and_type(array_shape, input_type)
    res = pymod_test_mod.ptrPyArrayObjectReturnArg(arg)
    assert numpy.all(res == arg)
    assert type(res) == type(arg)
    assert res.dtype == arg.dtype

@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_ptrPyArrayObjectReturnArg2(pymod_test_mod, random_1d_array_size, input_type):
    arg = array_utils.get_random_1d_array_of_size_and_type(random_1d_array_size, input_type)
    res = pymod_test_mod.ptrPyArrayObjectReturnArg(arg)
    assert numpy.all(res == arg)
    assert type(res) == type(arg)
    assert res.dtype == arg.dtype


def test_ptrPyArrayObjectReturnArg(pymod_test_mod):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.ptrPyArrayObjectReturnArg(arg)
    assert str(excinfo.value) == "argument 1 must be numpy.ndarray, not None"

