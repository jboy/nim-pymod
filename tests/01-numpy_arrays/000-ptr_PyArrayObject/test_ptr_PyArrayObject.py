import array_utils
import numpy
import pytest


def test_0_compile_pymod_test_mod(pmgen_py_compile):
        pmgen_py_compile(__name__)


@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_returnPyArrayObjectPtrAsInt(pymod_test_mod, random_1d_array_size, input_type):
    arg = numpy.zeros(random_1d_array_size, dtype=input_type)
    res = pymod_test_mod.returnPyArrayObjectPtrAsInt(arg)
    assert res == id(arg)


@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_ptrPyArrayObjectReturnArg(pymod_test_mod, random_1d_array_size, input_type):
    arg = numpy.arange(17)
    res = pymod_test_mod.ptrPyArrayObjectReturnArg(arg)
    assert numpy.all(res == arg)
    assert type(res) == type(arg)
    assert res.dtype == arg.dtype


def test_ptrPyArrayObjectReturnArg(pymod_test_mod):
    arg = None
    with pytest.raises(TypeError) as excinfo:
        pymod_test_mod.ptrPyArrayObjectReturnArg(arg)
    assert str(excinfo.value) == "argument 1 must be numpy.ndarray, not None"

