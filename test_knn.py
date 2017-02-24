import numpy as np
from l2_distance import l2_distance
from utils import *
from plot_digits import *
from run_knn import *
import matplotlib.pyplot as plt

"""load data
train_set = load_train()
valid_set = load_valid()
train_data = train_set[0]
train_labels = train_set[1]
valid_data = valid_set[0]
real_valid_label = valid_set[1]
"""
#train_data,train_label = load_train()
train_data,train_label = load_train_small()
#valid_data, real_valid_label = load_valid()
valid_data, real_valid_label = load_test()

k = [1,3,5,7,9]
for i in k:
    valid_label = run_knn(i, train_data, train_label, valid_data)
    correct_prediction = (valid_label == real_valid_label).astype(int)

    training_rate = float(np.sum(correct_prediction))/float(len(valid_label))
    plt.scatter(i,training_rate)
    plt.xlabel("K")
    plt.ylabel("Classification Rate")

plt.show()
