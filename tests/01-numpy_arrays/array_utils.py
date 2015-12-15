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


# 1-D arrays

def get_random_1d_array_of_size_and_type(test_1d_array_size, test_type):
    """Return a randomly-sized 1-D array of random integers in the range [-100, 100]."""
    return numpy.random.random_integers(-100, 100, test_1d_array_size
            ).astype(test_type)

# N-D arrays

def get_random_Nd_array_of_shape_and_type(test_Nd_array_shape, test_type):
    """Return a randomly-shaped N-D array of random integers in the range [-100, 100]."""
    array_size = numpy.prod(test_Nd_array_shape)
    return numpy.random.random_integers(-100, 100, array_size).reshape(test_Nd_array_shape
            ).astype(test_type)


def get_random_Nd_array_shape(ndim):
    if ndim == 1:
        dims = [numpy.random.randint(20) + 1]  # So the size is always >= 1.
    else:
        dims = [
                numpy.random.randint(10) + 1  # So the size is always >= 1.
                for i in range(ndim)]

    return tuple(dims)


def get_random_Nd_array_of_ndim_and_type(test_ndim, test_type):
    """Return a randomly-shaped N-D array of random integers in the range [-100, 100]."""
    array_shape = get_random_Nd_array_shape(test_ndim)
    array_size = numpy.prod(array_shape)
    return numpy.random.random_integers(-100, 100, array_size).reshape(array_shape
            ).astype(test_type)

