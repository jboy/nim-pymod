import numpy


def get_random_1d_array_of_type(test_1d_array_size, test_type):
    """Return a randomly-sized 1-D array of random integers in the range [-100, 100]."""
    return numpy.random.random_integers(-100, 100, test_1d_array_size
            ).astype(test_type)


def get_random_2d_array_of_type(test_2d_array_shape, test_type):
    """Return a randomly-shaped 2-D array of random integers in the range [-100, 100]."""
    array_size = numpy.prod(test_2d_array_shape)
    return numpy.random.random_integers(-100, 100, array_size).reshape(test_2d_array_shape
            ).astype(test_type)


def get_random_Nd_array_of_type(test_Nd_array_shape, test_type):
    """Return a randomly-shaped N-D array of random integers in the range [-100, 100]."""
    array_size = numpy.prod(test_Nd_array_shape)
    return numpy.random.random_integers(-100, 100, array_size).reshape(test_Nd_array_shape
            ).astype(test_type)

