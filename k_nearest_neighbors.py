##pr11
from __future__ import print_function
from __future__ import division
import math
import random

def convert_csv_to_data(fname):
    fh = open(fname)
    data = fh.read().replace("\r","\n").replace("\n\n", "\n").split("\n")
    fh.close()
    data_list = [[float(v) for v in d.replace("\n","").split(",")] for d in data if d]
    return data_list

fname = "pr11_concrete.csv"
data = convert_csv_to_data(fname)
print(data[0])

normal_data = []
avgs = [sum(d[i] for d in data)/len(data) for i in range(len(data[0]))]
normal_data = [[(vec[i] - avgs[i])/avgs[i] for i in range(len(vec))] for vec in data]
print(normal_data[0])

def calc_distance(vec1, vec2):
    return math.sqrt(sum((vec1[i] - vec2[i]) ** 2 for i in range(1, len(vec1))))

def get_naive_knn(data, vector, k=4):
    distances = [(calc_distance(vec, vector), vec) for vec in data]
    distances.sort()
    return distances[:k]

def get_knn(data, vector, k=4):
    k_nearest = []
    for vec in data:
        dist = calc_distance(vec, vector)
        if len(k_nearest) < k or dist < k_nearest[-1][0]:
            k_nearest = sorted(k_nearest + [(dist, vec)])[:k]
    return k_nearest

def calc_knn_output(data, vector, k=4):
    distance_vectors = get_knn(data, vector, k)
    return sum(vec[0] for dist, vec in distance_vectors)/len(distance_vectors)

def n_fold_cross_validate(all_data, n=10, k=4):
    errors = []
    for i in range(n):
        shuffled = all_data[:] ## gotta copy, because shuffles in place
        random.shuffle(shuffled)
        ntrain = int(len(shuffled) * (n-1) / n)
        training = shuffled[:ntrain]
        testing = shuffled[ntrain:]
        errors.append(math.sqrt(sum((vec[0] - calc_knn_output(training, vec, k)) ** 2 for vec in testing)))
    avg_error = sum(errors)/n
    return (avg_error, errors)

def find_best_knn(all_data, n=10, maxk=8):
    k_error_tuples = []
    for k in range(1, maxk+1):  ## 0 nearest neighbors makes no sense, so start at 1
        avg_error, errors = n_fold_cross_validate(all_data, n, k)
        print("For k={}, avg error was {} (all errors: {})\n\n".format(k, avg_error, errors))
        k_error_tuples.append((avg_error, k))
    least_error = min(k_error_tuples)
    print("Least error found when k={e[1]}, with avg error {e[0]}".format(e=least_error))
    return k_error_tuples

k = find_best_knn(normal_data, 10, 4)

