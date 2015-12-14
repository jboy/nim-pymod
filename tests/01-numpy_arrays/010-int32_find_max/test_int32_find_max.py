import array_utils
import numpy
import pytest


def test_0_compile_pymod_test_mod(pmgen_py_compile):
        pmgen_py_compile(__name__)


ndims_to_test = [1, 2, 3, 4]

@pytest.mark.parametrize("ndim", ndims_to_test)
def test_int32FindMaxForLoopValues(pymod_test_mod, seeded_random_number_generator, ndim):
    arg = array_utils.get_random_Nd_array_of_ndim_and_type(ndim, numpy.int32)
    print ("arg = %s" % arg)
    expectedRes = arg.max()
    res = pymod_test_mod.int32FindMaxForLoopValues(arg)
    assert res == expectedRes

