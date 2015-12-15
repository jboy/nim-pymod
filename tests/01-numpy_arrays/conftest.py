import numpy
import pytest
import time

from array_utils import get_range_of_random_integers
from array_utils import get_random_1d_array_size, get_random_Nd_array_dim
from array_utils import get_random_1d_array_of_size_and_type
from array_utils import get_random_Nd_array_shape
from array_utils import get_random_Nd_array_of_shape_and_type


@pytest.fixture(scope="module")
def seeded_random_number_generator():
    """Ensure that the random number generator has been seeded."""
    seed = int(time.time())
    numpy.random.seed(seed)
    return seed


# 1-D arrays

@pytest.fixture
def random_1d_array_size(seeded_random_number_generator):
    """Return a random integer in the range [1, 20] to be the size of a 1-D array."""
    return get_random_1d_array_size()

@pytest.fixture
def random_1d_array_of_bool(random_1d_array_size):
    """Return a randomly-sized array of random bool values."""
    return get_random_1d_array_of_size_and_type(random_1d_array_size, numpy.bool)

@pytest.fixture
def random_1d_array_of_integers(random_1d_array_size):
    """Return a randomly-sized 1-D array of random integers in the range [-100, 100]."""
    return get_random_1d_array_of_size_and_type(random_1d_array_size)


# 2-D arrays

@pytest.fixture
def random_2d_array_shape(seeded_random_number_generator):
    """Return a tuple of 2 random integers in the range [1, 10] to be the shape of a 2-D array."""
    return get_random_Nd_array_shape(2)

@pytest.fixture
def random_2d_array_of_integers(random_2d_array_shape):
    """Return a randomly-shaped 2-D array of random integers in the range [-100, 100]."""
    return get_random_Nd_array_of_shape_and_type(random_2d_array_shape)


# 3-D arrays

@pytest.fixture
def random_3d_array_shape(seeded_random_number_generator):
    """Return a tuple of 3 random integers in the range [1, 10] to be the shape of a 3-D array."""
    return get_random_Nd_array_shape(3)


# 4-D arrays

@pytest.fixture
def random_4d_array_shape(seeded_random_number_generator):
    """Return a tuple of 4 random integers in the range [1, 10] to be the shape of a 4-D array."""
    return get_random_Nd_array_shape(4)

