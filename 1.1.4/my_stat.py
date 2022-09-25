def array_mean(array):
    return sum([array[i] for i in range(array.size)]) / array.size

def standart_dev(array):
    mean_of_array = array_mean(array)
    return (sum([(array[i] - mean_of_array) ** 2 for i in range(array.size)]) / (array.size - 1)) ** 0.5

def data_in_sigma(data, sigma, n_sigma):
    data_mean = array_mean(data)
    counter = 0
    for i in range(data.size):
        if (abs(data[i] - data_mean) <= n_sigma * sigma):
            counter += 1
    return counter
