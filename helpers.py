import math
from decimal import Decimal, getcontext
import numpy

def manual_sort(column_data):
    if len(column_data) <= 1:
        return column_data
    else:
        pivot = column_data[0]
        left = []
        right = []
        for i in range(1, len(column_data)):
            if column_data[i] < pivot:
                left.append(column_data[i])
            else:
                right.append(column_data[i])
        return manual_sort(left) + [pivot] + manual_sort(right)

def pow(base, exponent):
    if exponent < 0:
        raise ValueError("pow() not defined for negative exponents")
    result = 1
    for _ in range(exponent):
        result *= base
    return result

def abs(x):
    return x if x >= 0 else -x

def sqrt(x):
    if x < 0:
        raise ValueError("sqrt() not defined for negative values")
    elif x == 0:
        return 0
    else:
        guess = x
        while True:
            new_guess = Decimal(0.5) * Decimal(guess + x / guess)
            if abs(guess - new_guess) < 1e-9:  # :sadge:
                return new_guess
            guess = new_guess


def count(column_data):
    num_values = 0
    for value in column_data:
        if not math.isnan(value):
            num_values += 1
    return num_values

def mean(column_data):
    num_values = count(column_data)
    if num_values == 0:
        return numpy.nan
    sum = Decimal(0)
    for value in column_data:
        if not math.isnan(value):
            sum += Decimal(value)
    return Decimal(sum / Decimal(num_values))

def std(column_data):
    num_values = count(column_data)
    if num_values == 0:
        return numpy.nan
    _mean = mean(column_data)
    getcontext().prec = 32  # :sadge:
    sum = Decimal(0)
    for value in column_data:
        if not math.isnan(value):
            sum += Decimal((Decimal(value) - _mean) ** 2)
    return sqrt(sum / Decimal(num_values - 1))


def percentile(data, percentile):
    data = [elem for elem in data if not math.isnan(elem)]
    if not data:
        return numpy.nan
    size = len(data)
    sorted_data = sorted(data)
    index = (percentile/100) * (size-1)
    if isinstance(index, int):
        return sorted_data[index]
    else:
        lower = int(index)
        upper = lower + 1
        return sorted_data[lower] * (upper - index) + sorted_data[upper] * (index - lower)

def min(column_data):
    clean_data = [elem for elem in column_data if not math.isnan(elem)]
    if clean_data:
        return clean_data[0]
    return numpy.nan

def max(column_data):
    clean_data = [elem for elem in column_data if not math.isnan(elem)]
    if clean_data:
        return clean_data[-1]
    return numpy.nan

def missing_values(column_data):
    return len([elem for elem in column_data if math.isnan(elem)])

def mode(column_data):
    clean_data = [elem for elem in column_data if not math.isnan(elem)]
    if not clean_data:
        return numpy.nan
    counts = {}
    for elem in clean_data:
        if elem in counts:
            counts[elem] += 1
        else:
            counts[elem] = 1
    max_count = max(manual_sort(list(set(counts.values())))) # :sadge:
    modes = [key for key, value in counts.items() if value == max_count]
    return min(modes)

def _range(min, max):
    if any(math.isnan(elem) for elem in [min, max]):
        return numpy.nan
    return max - min
