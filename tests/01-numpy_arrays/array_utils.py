# The utility functions in this module exist (ie, as non-fixtures) to enable us
# to parametrize test functions (eg, by ndim, shape or numpy type) and then use
# one of these utility functions within the test function to create the desired
# array.
#
# (AFAICT, you can't supply the params of a parametrized test function into the
# fixtures for that function.)
#
# These utility functions also enable us to specify just an ndim or shape value
# as the fixture for a test function, then create the necessary array (using
# that ndim or shape value) within the test function.  Often, the ndim or shape
# is all we really care about in the test; not the specific randomly-generated
# values in the array.


import numpy

all_supported_numpy_types = [
        numpy.bool,
        numpy.bool_,
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
]

all_supported_numpy_types_as_strings = [
        # NOTE:  Pymod returns the type string "<type 'numpy.bool'>" rather than "<type 'bool'>",
        # which is what `str(numpy.bool)` returns in Python.
        # Pymod returns "<type 'numpy.bool'>" for consistency with all other Numpy type strings,
        # which are all of the form "<type 'numpy.xxxx'>" (eg, "<type 'numpy.int8'>").
        (numpy.bool,    "numpy.bool"),
        (numpy.bool_,   "numpy.bool"),
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
]


# 0-dimensional arrays: eg, zd = numpy.array(17)
# zd.ndim == 0
# zd.shape == ()
# zd.size == 1
# zd[0] -> IndexError: too many indices for array
all_0d_array_shapes = [()]

# 0-size arrays: eg, zs = numpy.array([])
# zs.ndim == 1
# zs.shape == (0,)
# zs.size == 0
# zs[0] -> IndexError: index 0 is out of bounds for axis 0 with size 0
zero_size_1d_array_shapes = [(0,)]

zero_size_2d_array_shapes = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)]

zero_size_3d_array_shapes = [
        (0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0),
        (0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0),
        (0, 0, 2), (0, 2, 0), (2, 0, 0), (0, 2, 2), (2, 0, 2), (2, 2, 0)]

small_zero_size_array_shapes = (zero_size_1d_array_shapes +
        zero_size_2d_array_shapes + zero_size_3d_array_shapes)

small_1d_array_shapes = [(1,), (2,), (3,), (4,)]

small_2d_array_shapes = [(1, 1), (1, 2), (2, 1), (2, 2), (1, 3), (3, 1), (1, 4), (4, 1)]

small_3d_array_shapes = [(1, 1, 1),
        (1, 1, 2), (1, 2, 1), (2, 1, 1), (1, 2, 2), (2, 2, 1), (2, 1, 2),
        (1, 1, 3), (1, 3, 1), (3, 1, 1), (1, 1, 4), (1, 4, 1), (4, 1, 1)]

small_4d_array_shapes = [(1, 1, 1, 1),
  (1, 1, 1, 2), (1, 1, 2, 1), (1, 2, 1, 1), (2, 1, 1, 1),
  (1, 1, 1, 3), (1, 1, 3, 1), (1, 3, 1, 1), (3, 1, 1, 1),
  (1, 1, 1, 4), (1, 1, 4, 1), (1, 4, 1, 1), (4, 1, 1, 1),
  (1, 1, 2, 2), (1, 2, 1, 2), (1, 2, 2, 1), (2, 1, 2, 1), (2, 2, 1, 1), (2, 1, 1, 2)]


indexable_small_array_shapes = (small_1d_array_shapes +
        small_2d_array_shapes + small_3d_array_shapes + small_4d_array_shapes)

nonzero_size_small_array_shapes = (all_0d_array_shapes +
        indexable_small_array_shapes)

nonzero_ndim_small_array_shapes = (small_zero_size_array_shapes +
        indexable_small_array_shapes)

all_small_array_shapes = (all_0d_array_shapes + small_1d_array_shapes +
        indexable_small_array_shapes)


def get_range_of_random_integers(target_type=None):
    if target_type is not None:
        if target_type in (numpy.bool, numpy.bool_):
            # Since target type is `bool`, return [0, 1] as the range, so that
            # there will be a more balanced distribution between True & False.
            return (0, 1)

    return (-100, 100)  # Small enough to fit into a positive int8.


def get_random_1d_array_size():
    # This is 2x the per-dimension size for 2+-D arrays.
    return numpy.random.randint(20) + 1  # So the size is always >= 1.

def get_random_Nd_array_dim():
    return numpy.random.randint(10) + 1  # So the dimension is always >= 1.


# 1-D arrays

def get_random_1d_array_of_size_and_type(test_1d_array_size, test_type=None):
    (lower, upper) = get_range_of_random_integers(test_type)
    arr = numpy.random.random_integers(lower, upper, test_1d_array_size)
    return arr if test_type is None else arr.astype(test_type)

# N-D arrays

def get_random_Nd_array_shape(ndim):
    if ndim == 1:
        return (get_random_1d_array_size(),)
    else:
        return tuple([get_random_Nd_array_dim() for i in range(ndim)])


def get_random_Nd_array_of_shape_and_type(test_Nd_array_shape, test_type=None):
    array_size = numpy.prod(test_Nd_array_shape)
    (lower, upper) = get_range_of_random_integers(test_type)
    arr = numpy.random.random_integers(lower, upper, array_size).reshape(test_Nd_array_shape)
    return arr if test_type is None else arr.astype(test_type)


def get_random_Nd_array_of_ndim_and_type(test_ndim, test_type=None):
    test_array_shape = get_random_Nd_array_shape(test_ndim)
    return get_random_Nd_array_of_shape_and_type(test_array_shape, test_type)

