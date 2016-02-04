import array_utils
import numpy
import pytest


def test_0_compile_pymod_test_mod(pmgen_py_compile):
        pmgen_py_compile(__name__)


ndims_to_test = [1, 2, 3, 4]


# for loop, values

@pytest.mark.parametrize("ndim", ndims_to_test)
@pytest.mark.parametrize("nim_test_proc_name", [
        "int32FindMaxForLoopValues",
        "int32FindMaxForLoopValues_m",
])
def test_int32FindMaxForLoopValues(pymod_test_mod, seeded_random_number_generator,
        ndim, nim_test_proc_name):
    arg = array_utils.get_random_Nd_array_of_ndim_and_type(ndim, numpy.int32)
    print ("\nrandom number seed = %d\nndim = %d, shape = %s\narg =\n%s" % \
            (seeded_random_number_generator, ndim, arg.shape, arg))
    expectedRes = arg.max()
    res = getattr(pymod_test_mod, nim_test_proc_name)(arg)
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
@pytest.mark.parametrize("nim_test_proc_name", [
        "int32FindMaxForLoopForwardIter",
        "int32FindMaxForLoopForwardIter_m",
        "int32FindMaxForLoopForwardIter_i",
])
def test_int32FindMaxForLoopForwardIter(pymod_test_mod, seeded_random_number_generator,
        ndim, nim_test_proc_name):
    arg = array_utils.get_random_Nd_array_of_ndim_and_type(ndim, numpy.int32)
    print ("\nnim_test_proc_name = %s" % nim_test_proc_name)
    print ("random number seed = %d\nndim = %d, shape = %s\narg =\n%s" % \
            (seeded_random_number_generator, ndim, arg.shape, arg))
    expectedRes = arg.max()
    res = getattr(pymod_test_mod, nim_test_proc_name)(arg)
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
def test_int32FindMaxWhileLoopRandaccIterDerefKParamAlternatives(pymod_test_mod, seeded_random_number_generator,
        ndim, nim_test_proc_name, k):
    arg = array_utils.get_random_Nd_array_of_ndim_and_type(ndim, numpy.int32)
    print ("\nnim_test_proc_name = %s, k = %d" % (nim_test_proc_name, k))
    print ("random number seed = %d\nndim = %d, shape = %s\narg =\n%s" % \
            (seeded_random_number_generator, ndim, arg.shape, arg))
    expectedRes = arg.max()
    res = getattr(pymod_test_mod, nim_test_proc_name)(arg, k)
    print ("res = %s" % str(res))
    assert res == expectedRes


@pytest.mark.parametrize("ndim", ndims_to_test)
@pytest.mark.parametrize("nim_test_proc_name", [
        "int32FindMaxWhileLoopRandaccIterDeltaN_1",
        "int32FindMaxWhileLoopRandaccIterDeltaN_2",
])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 10, 100, 1000])
def test_int32FindMaxWhileLoopRandaccIterDeltaN_1(pymod_test_mod, seeded_random_number_generator,
        ndim, nim_test_proc_name, n):
    arg = array_utils.get_random_Nd_array_of_ndim_and_type(ndim, numpy.int32)
    print ("\nnim_test_proc_name = %s, n = %d" % (nim_test_proc_name, n))
    print ("random number seed = %d\nndim = %d, shape = %s\narg =\n%s" % \
            (seeded_random_number_generator, ndim, arg.shape, arg))
    argDeltaN = arg.flat[::n]
    print ("arg.flat[::n] =\n%s" % argDeltaN)
    expectedRes = argDeltaN.max()
    res = getattr(pymod_test_mod, nim_test_proc_name)(arg, n)
    print ("res = %s" % str(res))
    assert res == expectedRes


@pytest.mark.parametrize("ndim", ndims_to_test)
@pytest.mark.parametrize("nim_test_proc_name", [
        "int32FindMaxWhileLoopRandaccIterExcludeFirstM_1",
        "int32FindMaxWhileLoopRandaccIterExcludeFirstM_2",
])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 10, 100, 1000])
def test_int32FindMaxWhileLoopRandaccIterExcludeFirstM_1(pymod_test_mod, seeded_random_number_generator,
        ndim, nim_test_proc_name, m):
    dtype = numpy.int32
    arg = array_utils.get_random_Nd_array_of_ndim_and_type(ndim, dtype)
    print ("\nnim_test_proc_name = %s, m = %d" % (nim_test_proc_name, m))
    print ("random number seed = %d\nndim = %d, shape = %s\narg =\n%s" % \
            (seeded_random_number_generator, ndim, arg.shape, arg))
    argAfterM = arg.flat[m:]
    print ("arg.flat[m:] =\n%s" % argAfterM)
    if argAfterM.size > 0:
        expectedRes = argAfterM.max()
        print ("expectedRes = %s" % str(expectedRes))
    else:
        expectedRes = numpy.iinfo(dtype).min
        print ("expectedRes = %s  (int32.min)" % str(expectedRes))
    res = getattr(pymod_test_mod, nim_test_proc_name)(arg, m)
    print ("res = %s" % str(res))
    assert res == expectedRes


@pytest.mark.parametrize("ndim", ndims_to_test)
@pytest.mark.parametrize("nim_test_proc_name", [
        "int32FindMaxWhileLoopRandaccIterExcludeLastM_1",
        "int32FindMaxWhileLoopRandaccIterExcludeLastM_2",
])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 10, 100, 1000])
def test_int32FindMaxWhileLoopRandaccIterExcludeLastM_1(pymod_test_mod, seeded_random_number_generator,
        ndim, nim_test_proc_name, m):
    dtype = numpy.int32
    arg = array_utils.get_random_Nd_array_of_ndim_and_type(ndim, dtype)
    print ("\nnim_test_proc_name = %s, m = %d" % (nim_test_proc_name, m))
    print ("random number seed = %d\nndim = %d, shape = %s\narg =\n%s" % \
            (seeded_random_number_generator, ndim, arg.shape, arg))
    argBeforeLastM = arg.flat[:-m]
    print ("arg.flat[:-m] =\n%s" % argBeforeLastM)
    if argBeforeLastM.size > 0:
        expectedRes = argBeforeLastM.max()
        print ("expectedRes = %s" % str(expectedRes))
    else:
        expectedRes = numpy.iinfo(dtype).min
        print ("expectedRes = %s  (int32.min)" % str(expectedRes))
    res = getattr(pymod_test_mod, nim_test_proc_name)(arg, m)
    print ("res = %s" % str(res))
    assert res == expectedRes


# for loop, Rand Acc Iter

@pytest.mark.parametrize("ndim", ndims_to_test)
@pytest.mark.parametrize("nim_test_proc_name", [
        "int32FindMaxForLoopRandaccIterDeref",
        "int32FindMaxForLoopRandaccIterDeref_m",
        "int32FindMaxForLoopRandaccIterDeref_i",
        "int32FindMaxForLoopRandaccIterIndex0_i",
])
def test_int32FindMaxForLoopRandaccIterDerefAlternatives(pymod_test_mod, seeded_random_number_generator,
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
        "int32FindMaxForLoopRandaccIterDeltaN",
        "int32FindMaxForLoopRandaccIterDeltaN_m",
        "int32FindMaxForLoopRandaccIterDeltaN_i",
])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 10, 100, 1000])
def test_int32FindMaxForLoopRandaccIterDeltaN_1(pymod_test_mod, seeded_random_number_generator,
        ndim, nim_test_proc_name, n):
    arg = array_utils.get_random_Nd_array_of_ndim_and_type(ndim, numpy.int32)
    print ("\nnim_test_proc_name = %s, n = %d" % (nim_test_proc_name, n))
    print ("random number seed = %d\nndim = %d, shape = %s\narg =\n%s" % \
            (seeded_random_number_generator, ndim, arg.shape, arg))
    argDeltaN = arg.flat[::n]
    print ("arg.flat[::n] =\n%s" % argDeltaN)
    expectedRes = argDeltaN.max()
    res = getattr(pymod_test_mod, nim_test_proc_name)(arg, n)
    print ("res = %s" % str(res))
    assert res == expectedRes


@pytest.mark.parametrize("ndim", ndims_to_test)
@pytest.mark.parametrize("nim_test_proc_name", [
        "int32FindMaxForLoopRandaccIterExcludeFirstM",
        "int32FindMaxForLoopRandaccIterExcludeFirstM_m",
        "int32FindMaxForLoopRandaccIterExcludeFirstM_i",
])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 10, 100, 1000])
def test_int32FindMaxForLoopRandaccIterExcludeFirstM_1(pymod_test_mod, seeded_random_number_generator,
        ndim, nim_test_proc_name, m):
    dtype = numpy.int32
    arg = array_utils.get_random_Nd_array_of_ndim_and_type(ndim, dtype)
    print ("\nnim_test_proc_name = %s, m = %d" % (nim_test_proc_name, m))
    print ("random number seed = %d\nndim = %d, shape = %s\narg =\n%s" % \
            (seeded_random_number_generator, ndim, arg.shape, arg))
    argAfterM = arg.flat[m:]
    print ("arg.flat[m:] =\n%s" % argAfterM)
    if argAfterM.size > 0:
        expectedRes = argAfterM.max()
        print ("expectedRes = %s" % str(expectedRes))
    else:
        expectedRes = numpy.iinfo(dtype).min
        print ("expectedRes = %s  (int32.min)" % str(expectedRes))
    res = getattr(pymod_test_mod, nim_test_proc_name)(arg, m)
    print ("res = %s" % str(res))
    assert res == expectedRes


@pytest.mark.parametrize("ndim", ndims_to_test)
@pytest.mark.parametrize("nim_test_proc_name", [
        "int32FindMaxForLoopRandaccIterExcludeLastM_i",
])
@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 10, 100, 1000])
def test_int32FindMaxForLoopRandaccIterExcludeLastM_1(pymod_test_mod, seeded_random_number_generator,
        ndim, nim_test_proc_name, m):
    dtype = numpy.int32
    arg = array_utils.get_random_Nd_array_of_ndim_and_type(ndim, dtype)
    print ("\nnim_test_proc_name = %s, m = %d" % (nim_test_proc_name, m))
    print ("random number seed = %d\nndim = %d, shape = %s\narg =\n%s" % \
            (seeded_random_number_generator, ndim, arg.shape, arg))
    argBeforeLastM = arg.flat[:-m]
    print ("arg.flat[:-m] =\n%s" % argBeforeLastM)
    if argBeforeLastM.size > 0:
        expectedRes = argBeforeLastM.max()
        print ("expectedRes = %s" % str(expectedRes))
    else:
        expectedRes = numpy.iinfo(dtype).min
        print ("expectedRes = %s  (int32.min)" % str(expectedRes))
    res = getattr(pymod_test_mod, nim_test_proc_name)(arg, m)
    print ("res = %s" % str(res))
    assert res == expectedRes

