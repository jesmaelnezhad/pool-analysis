def average_on_columns(tuples):
    """
    Averages the tuple values on columns
    :param tuples: list of same size tuples
    :return: a tuple with the same size containing the averages
    """
    a_tuple = tuples[0]
    return tuple(sum(t[t_index] for t in tuples) / len(tuples) for t_index in range(len(a_tuple)))


def min_on_columns(tuples):
    """
    Finds the minimum of the tuple values on columns
    :param tuples: list of same size tuples
    :return: a tuple with the same size containing the minimums
    """
    a_tuple = tuples[0]
    return tuple(min(t[t_index] for t in tuples) for t_index in range(len(a_tuple)))


def max_on_columns(tuples):
    """
    Finds the maximum of the tuple values on columns
    :param tuples: list of same size tuples
    :return: a tuple with the same size containing the maximums
    """
    a_tuple = tuples[0]
    return tuple(max(t[t_index] for t in tuples) for t_index in range(len(a_tuple)))
