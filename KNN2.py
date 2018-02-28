import numpy as np
from math import sqrt
import warnings
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}

new_feature = [8, 4]
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feature[0], new_feature[1], s=100, color='blue')
plt.show()

def k_nearest_neighbours(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("dataset being too large")

    distances = []

    for group in data:
        for features in data[group]:
            #eucledian_dis = np.sqrt(np.sum((np.array(features) - np.array(predict)) ** 2))
            eucledian_dis = np.linalg.norm(np.array(features) - np.array(predict))

            distances.append([eucledian_dis, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common())

    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

result = k_nearest_neighbours(dataset, new_feature, k=3)

print(result)


[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feature[0], new_feature[1], s=100, color=result)
plt.show()










