import array_utils
import numpy
import pytest


def test_0_compile_pymod_test_mod(pmgen_py_compile):
        pmgen_py_compile(__name__)


@pytest.mark.parametrize("input_type,input_type_str", array_utils.all_supported_numpy_types_as_strings)
def test_returnDtypeAsString(pymod_test_mod, random_1d_array_size, input_type, input_type_str):
    arg = numpy.zeros(random_1d_array_size, dtype=input_type)
    res = pymod_test_mod.returnDtypeAsString(arg)
    assert res == input_type_str

