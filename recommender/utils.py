import numpy as np
from datetime import datetime, timedelta


def status_printer(num_steps, general_duration):
    """
    This function prints the duration of one process with #iterations=num_steps
    and duration = general_duration.
    """
    sec = timedelta(seconds=int(general_duration))
    d_time = datetime(1, 1, 1) + sec
    print(' ')
    print('The duration of the whole training with % s steps is %.2f seconds,'
          % (num_steps, general_duration))
    print("which is equal to:  %d:%d:%d:%d"
          % (d_time.day-1, d_time.hour, d_time.minute, d_time.second), end='')
    print(" (DAYS:HOURS:MIN:SEC)")


def mse(x, y):
    """
    Return the mean square error between the
    x and y arrays.


    :type x: numpy array
    :type y: numpy array
    :rtype: float
    """
    assert len(x) == len(y)
    return np.mean(np.power(x - y, 2))


def rmse(x, y):
    """
    Return the root-mean-square error between the
    x and y arrays.


    :type x: numpy array
    :type y: numpy array
    :rtype: float
    """
    return np.sqrt(mse(x, y))
