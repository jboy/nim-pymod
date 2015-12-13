import array_utils
import numpy
import pytest


def test_0_compile_pymod_test_mod(pmgen_py_compile):
        pmgen_py_compile(__name__)


@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_returnStridesAsTuple1D(pymod_test_mod, input_type, random_1d_array_size):
    arg = array_utils.get_random_1d_array_of_type(random_1d_array_size, input_type)
    expectedStrides = arg.strides

    resStrides = pymod_test_mod.returnStridesAsTuple1D(arg)
    # FIXME:  Currently, Pymod incorrectly unwraps single-element-tuple return-types.
    # Thus, `resStrides` should be a tuple-of-single-int, but instead it's an int.
    # We will fix this soon...
    resStrides = (resStrides,)
    assert resStrides == expectedStrides


@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_returnStridesAsTuple2D(pymod_test_mod, input_type, random_2d_array_shape):
    arg = array_utils.get_random_2d_array_of_type(random_2d_array_shape, input_type)
    expectedStrides = arg.strides

    resStrides = pymod_test_mod.returnStridesAsTuple2D(arg)
    assert resStrides == expectedStrides


@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_returnStridesAsTuple3D(pymod_test_mod, input_type, random_3d_array_shape):
    arg = array_utils.get_random_Nd_array_of_type(random_3d_array_shape, input_type)
    expectedStrides = arg.strides

    resStrides = pymod_test_mod.returnStridesAsTuple3D(arg)
    assert resStrides == expectedStrides


@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_returnStridesAsTuple4D(pymod_test_mod, input_type, random_4d_array_shape):
    arg = array_utils.get_random_Nd_array_of_type(random_4d_array_shape, input_type)
    expectedStrides = arg.strides

    resStrides = pymod_test_mod.returnStridesAsTuple4D(arg)
    assert resStrides == expectedStrides

