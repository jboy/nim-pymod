import numpy
import pytest
import time


@pytest.fixture(scope="module")
def seeded_random_number_generator():
    """Ensure that the random number generator has been seeded."""
    seed = int(time.time())
    numpy.random.seed(seed)
    return seed


_NUM_ELEMS_1D = 20  # 2x the per-dimension size for 2-D and larger.
_NUM_ELEMS_ND = 10

def _get_random_1d_array_size():
    return numpy.random.randint(_NUM_ELEMS_1D) + 1  # So the size is always >= 1.

def _get_random_Nd_array_dim():
    return numpy.random.randint(_NUM_ELEMS_ND) + 1  # So the dimension is always >= 1.

_RANDOM_INTEGER_RANGE = (-100, 100)  # Small enough to fit into a positive int8.


# 1-D arrays

@pytest.fixture
def random_1d_array_size(seeded_random_number_generator):
    """Return a random integer in the range [1, 20] to be the size of a 1-D array."""
    return _get_random_1d_array_size()

@pytest.fixture
def random_1d_array_of_bool(random_1d_array_size):
    """Return a randomly-sized array of random bool values."""
    return numpy.random.random_integers(0, 1, random_1d_array_size).astype(numpy.bool)

@pytest.fixture
def random_1d_array_of_integers(random_1d_array_size):
    """Return a randomly-sized 1-D array of random integers in the range [-100, 100]."""
    (lower, upper) = _RANDOM_INTEGER_RANGE
    return numpy.random.random_integers(lower, upper, random_1d_array_size)


# 2-D arrays

@pytest.fixture
def random_2d_array_shape(seeded_random_number_generator):
    """Return a tuple of 2 random integers in the range [1, 10] to be the shape of a 2-D array."""
    dim1 = _get_random_Nd_array_dim()
    dim2 = _get_random_Nd_array_dim()
    return (dim1, dim2)

@pytest.fixture
def random_2d_array_of_integers(random_2d_array_shape):
    """Return a randomly-shaped 2-D array of random integers in the range [-100, 100]."""
    array_size = numpy.prod(random_2d_array_shape)
    (lower, upper) = _RANDOM_INTEGER_RANGE
    return numpy.random.random_integers(lower, upper, array_size).reshape(random_2d_array_shape)


# 3-D arrays

@pytest.fixture
def random_3d_array_shape(seeded_random_number_generator):
    """Return a tuple of 3 random integers in the range [1, 10] to be the shape of a 3-D array."""
    dim1 = _get_random_Nd_array_dim()
    dim2 = _get_random_Nd_array_dim()
    dim3 = _get_random_Nd_array_dim()
    return (dim1, dim2, dim3)


# 4-D arrays

@pytest.fixture
def random_4d_array_shape(seeded_random_number_generator):
    """Return a tuple of 4 random integers in the range [1, 10] to be the shape of a 4-D array."""
    dim1 = _get_random_Nd_array_dim()
    dim2 = _get_random_Nd_array_dim()
    dim3 = _get_random_Nd_array_dim()
    dim4 = _get_random_Nd_array_dim()
    return (dim1, dim2, dim3, dim4)

