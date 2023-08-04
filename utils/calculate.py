import math


def margin_of_error(values, confidence_interval=1.96):
    num = len(values)
    mean = sum(values) / num
    variance = sum(list(map(lambda x: pow(x - mean, 2), values))) / num

    standard_deviation = math.sqrt(variance)
    standard_error = standard_deviation / math.sqrt(num)

    return mean, standard_error * confidence_interval
