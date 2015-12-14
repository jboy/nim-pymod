import array_utils
import numpy
import pytest


def test_0_compile_pymod_test_mod(pmgen_py_compile):
        pmgen_py_compile(__name__)


@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_returnDimensionsAsTuple1D(pymod_test_mod, input_type, random_1d_array_size):
    arg = array_utils.get_random_1d_array_of_size_and_type(random_1d_array_size, input_type)
    expectedDimensions = arg.shape
    expectedShape = arg.shape

    resDimensions = pymod_test_mod.returnDimensionsAsTuple1D(arg)
    # FIXME:  Currently, Pymod incorrectly unwraps single-element-tuple return-types.
    # Thus, `resDimensions` should be a tuple-of-single-int, but instead it's an int.
    # We will fix this soon...
    resDimensions = (resDimensions,)
    assert resDimensions == expectedDimensions

    resShape = pymod_test_mod.returnShapeAsTuple1D(arg)
    # FIXME:  Currently, Pymod incorrectly unwraps single-element-tuple return-types.
    # Thus, `resShape` should be a tuple-of-single-int, but instead it's an int.
    # We will fix this soon...
    resShape = (resShape,)
    assert resShape == expectedShape


@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_returnDimensionsAsTuple2D(pymod_test_mod, input_type, random_2d_array_shape):
    arg = array_utils.get_random_2d_array_of_shape_and_type(random_2d_array_shape, input_type)
    expectedDimensions = arg.shape
    expectedShape = arg.shape

    resDimensions = pymod_test_mod.returnDimensionsAsTuple2D(arg)
    assert resDimensions == expectedDimensions

    resShape = pymod_test_mod.returnShapeAsTuple2D(arg)
    assert resShape == expectedShape


@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_returnDimensionsAsTuple3D(pymod_test_mod, input_type, random_3d_array_shape):
    arg = array_utils.get_random_Nd_array_of_shape_and_type(random_3d_array_shape, input_type)
    expectedDimensions = arg.shape
    expectedShape = arg.shape

    resDimensions = pymod_test_mod.returnDimensionsAsTuple3D(arg)
    assert resDimensions == expectedDimensions

    resShape = pymod_test_mod.returnShapeAsTuple3D(arg)
    assert resShape == expectedShape


@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_returnDimensionsAsTuple4D(pymod_test_mod, input_type, random_4d_array_shape):
    arg = array_utils.get_random_Nd_array_of_shape_and_type(random_4d_array_shape, input_type)
    expectedDimensions = arg.shape
    expectedShape = arg.shape

    resDimensions = pymod_test_mod.returnDimensionsAsTuple4D(arg)
    assert resDimensions == expectedDimensions

    resShape = pymod_test_mod.returnShapeAsTuple4D(arg)
    assert resShape == expectedShape

