import array_utils
import numpy
import pytest


def test_0_compile_pymod_test_mod(pmgen_py_compile):
        pmgen_py_compile(__name__)


@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_returnNdAttr_1d(pymod_test_mod, input_type, random_1d_array_size):
    arg = array_utils.get_random_1d_array_of_size_and_type(random_1d_array_size, input_type)

    resNd = pymod_test_mod.returnNdAttr(arg)
    assert resNd == len(arg.shape)
    assert resNd == arg.ndim

    resNdim = pymod_test_mod.returnNdimAttr(arg)
    assert resNdim == len(arg.shape)
    assert resNdim == arg.ndim


@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_returnNdAttr_2d(pymod_test_mod, input_type, random_2d_array_shape):
    arg = array_utils.get_random_Nd_array_of_shape_and_type(random_2d_array_shape, input_type)

    resNd = pymod_test_mod.returnNdAttr(arg)
    assert resNd == len(arg.shape)
    assert resNd == arg.ndim

    resNdim = pymod_test_mod.returnNdimAttr(arg)
    assert resNdim == len(arg.shape)
    assert resNdim == arg.ndim


@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_returnNdAttr_3d(pymod_test_mod, input_type, random_3d_array_shape):
    arg = array_utils.get_random_Nd_array_of_shape_and_type(random_3d_array_shape, input_type)

    resNd = pymod_test_mod.returnNdAttr(arg)
    assert resNd == len(arg.shape)
    assert resNd == arg.ndim

    resNdim = pymod_test_mod.returnNdimAttr(arg)
    assert resNdim == len(arg.shape)
    assert resNdim == arg.ndim


@pytest.mark.parametrize("input_type", array_utils.all_supported_numpy_types)
def test_returnNdAttr_4d(pymod_test_mod, input_type, random_4d_array_shape):
    arg = array_utils.get_random_Nd_array_of_shape_and_type(random_4d_array_shape, input_type)

    resNd = pymod_test_mod.returnNdAttr(arg)
    assert resNd == len(arg.shape)
    assert resNd == arg.ndim

    resNdim = pymod_test_mod.returnNdimAttr(arg)
    assert resNdim == len(arg.shape)
    assert resNdim == arg.ndim

