import importlib
import numpy
import pytest
import time


@pytest.fixture(scope="module")
def seeded_random_number_generator():
    """Ensure that the random number generator has been seeded."""
    numpy.random.seed(int(time.time()))


@pytest.fixture
def random_1d_array_size(seeded_random_number_generator):
    """Return a random integer in the range [1, 20] to be the size of a 1-D array."""
    return numpy.random.randint(20) + 1  # So the size is always >= 1.

@pytest.fixture
def random_1d_array_of_bool(random_1d_array_size):
    """Return a randomly-sized array of random bool values."""
    return numpy.random.random_integers(0, 1, random_1d_array_size).astype(numpy.bool)

@pytest.fixture
def random_1d_array_of_integers(random_1d_array_size):
    """Return a randomly-sized 1-D array of random integers in the range [-100, 100]."""
    return numpy.random.random_integers(-100, 100, random_1d_array_size)


@pytest.fixture
def random_2d_array_shape(seeded_random_number_generator):
    """Return a tuple of 2 random integers in the range [1, 10] to be the shape of a 2-D array."""
    dim1 = numpy.random.randint(10) + 1  # So the size is always >= 1.
    dim2 = numpy.random.randint(10) + 1  # So the size is always >= 1.
    return (dim1, dim2)

@pytest.fixture
def random_2d_array_of_integers(random_2d_array_shape):
    """Return a randomly-shaped 2-D array of random integers in the range [-100, 100]."""
    array_size = numpy.prod(random_2d_array_shape)
    return numpy.random.random_integers(-100, 100, array_size).reshape(random_2d_array_shape)


@pytest.fixture
def random_3d_array_shape(seeded_random_number_generator):
    """Return a tuple of 3 random integers in the range [1, 10] to be the shape of a 3-D array."""
    dim1 = numpy.random.randint(10) + 1  # So the size is always >= 1.
    dim2 = numpy.random.randint(10) + 1  # So the size is always >= 1.
    dim3 = numpy.random.randint(10) + 1  # So the size is always >= 1.
    return (dim1, dim2, dim3)

@pytest.fixture
def random_4d_array_shape(seeded_random_number_generator):
    """Return a tuple of 4 random integers in the range [1, 10] to be the shape of a 4-D array."""
    dim1 = numpy.random.randint(10) + 1  # So the size is always >= 1.
    dim2 = numpy.random.randint(10) + 1  # So the size is always >= 1.
    dim3 = numpy.random.randint(10) + 1  # So the size is always >= 1.
    dim4 = numpy.random.randint(10) + 1  # So the size is always >= 1.
    return (dim1, dim2, dim3, dim4)


#@pytest.fixture
#def array_utils():
#    return importlib.import_module("array_utils_lib")
