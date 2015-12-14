import array_utils
import numpy
import pytest


def test_0_compile_pymod_test_mod(pmgen_py_compile):
        pmgen_py_compile(__name__)


ndims_to_test = [1, 2, 3, 4]


# for loop, values

@pytest.mark.parametrize("ndim", ndims_to_test)
def test_int32FindMaxForLoopValues(pymod_test_mod, seeded_random_number_generator, ndim):
    arg = array_utils.get_random_Nd_array_of_ndim_and_type(ndim, numpy.int32)
    print ("\nrandom number seed = %d\nndim = %d, shape = %s\narg =\n%s" % \
            (seeded_random_number_generator, ndim, arg.shape, arg))
    expectedRes = arg.max()
    res = pymod_test_mod.int32FindMaxForLoopValues(arg)
    print ("res = %s" % str(res))
    assert res == expectedRes


# while loop, Forward Iter

@pytest.mark.parametrize("ndim", ndims_to_test)
def test_int32FindMaxWhileLoopForwardIter(pymod_test_mod, seeded_random_number_generator, ndim):
    arg = array_utils.get_random_Nd_array_of_ndim_and_type(ndim, numpy.int32)
    print ("\nrandom number seed = %d\nndim = %d, shape = %s\narg =\n%s" % \
            (seeded_random_number_generator, ndim, arg.shape, arg))
    expectedRes = arg.max()
    res = pymod_test_mod.int32FindMaxWhileLoopForwardIter(arg)
    print ("res = %s" % str(res))
    assert res == expectedRes


# for loop, Forward Iter

@pytest.mark.parametrize("ndim", ndims_to_test)
def test_int32FindMaxForLoopForwardIter(pymod_test_mod, seeded_random_number_generator, ndim):
    arg = array_utils.get_random_Nd_array_of_ndim_and_type(ndim, numpy.int32)
    print ("\nrandom number seed = %d\nndim = %d, shape = %s\narg =\n%s" % \
            (seeded_random_number_generator, ndim, arg.shape, arg))
    expectedRes = arg.max()
    res = pymod_test_mod.int32FindMaxForLoopForwardIter(arg)
    print ("res = %s" % str(res))
    assert res == expectedRes


# while loop, Rand Acc Iter

@pytest.mark.parametrize("ndim", ndims_to_test)
@pytest.mark.parametrize("nim_test_proc_name", [
        "int32FindMaxWhileLoopRandaccIterDeref",
        "int32FindMaxWhileLoopRandaccIterIndex0",
        "int32FindMaxWhileLoopRandaccIterDerefPlusZeroOffset",
        "int32FindMaxWhileLoopRandaccIterDerefMinusZeroOffset",
        "int32FindMaxWhileLoopRandaccIterIndexVsPlusOffset_1",
        "int32FindMaxWhileLoopRandaccIterIndexVsPlusOffset_2",
        "int32FindMaxWhileLoopRandaccIterIndexVsPlusOffset_3",
        "int32FindMaxWhileLoopRandaccIterIndexVsPlusOffset_4",
        "int32FindMaxWhileLoopRandaccIterIndexVsPlusOffset_5",
        "int32FindMaxWhileLoopRandaccIterIndexVsMinusOffset_1",
        "int32FindMaxWhileLoopRandaccIterIndexVsMinusOffset_2",
        "int32FindMaxWhileLoopRandaccIterIndexVsMinusOffset_3",
        "int32FindMaxWhileLoopRandaccIterIndexVsMinusOffset_4",
        "int32FindMaxWhileLoopRandaccIterIndexVsMinusOffset_5",
])
def test_int32FindMaxWhileLoopRandaccIterDerefAlternatives(pymod_test_mod, seeded_random_number_generator,
        ndim, nim_test_proc_name):
    arg = array_utils.get_random_Nd_array_of_ndim_and_type(ndim, numpy.int32)
    print ("\nnim_test_proc_name = %s" % nim_test_proc_name)
    print ("random number seed = %d\nndim = %d, shape = %s\narg =\n%s" % \
            (seeded_random_number_generator, ndim, arg.shape, arg))
    expectedRes = arg.max()
    res = getattr(pymod_test_mod, nim_test_proc_name)(arg)
    print ("res = %s" % str(res))
    assert res == expectedRes


@pytest.mark.parametrize("ndim", ndims_to_test)
@pytest.mark.parametrize("nim_test_proc_name", [
        "int32FindMaxWhileLoopRandaccIterIndexVsPlusOffsetK",
        "int32FindMaxWhileLoopRandaccIterIndexVsMinusOffsetK",
])
@pytest.mark.parametrize("k", [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
def test_int32FindMaxWhileLoopRandaccIterDerefAlternatives(pymod_test_mod, seeded_random_number_generator,
        ndim, nim_test_proc_name, k):
    arg = array_utils.get_random_Nd_array_of_ndim_and_type(ndim, numpy.int32)
    print ("\nnim_test_proc_name = %s, k = %d" % (nim_test_proc_name, k))
    print ("random number seed = %d\nndim = %d, shape = %s\narg =\n%s" % \
            (seeded_random_number_generator, ndim, arg.shape, arg))
    expectedRes = arg.max()
    res = getattr(pymod_test_mod, nim_test_proc_name)(arg, k)
    print ("res = %s" % str(res))
    assert res == expectedRes

