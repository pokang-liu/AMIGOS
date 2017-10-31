import numpy as np


def fisher(features, labels):
    Fisher = []
    labels = np.array(labels)
    labels0 = np.where(labels > 0)
    labels1 = np.where(labels < 1)
    labels0 = np.array(labels0)
    features0 = np.delete(features, labels0, axis=0)
    features1 = np.delete(features, labels1, axis=0)

    mean_features0 = np.mean(features0, axis=0)
    mean_features1 = np.mean(features1, axis=0)

    std_features0 = np.std(features0)
    std_features1 = np.std(features1)
    std_sum = (std_features1) * (std_features1) + std_features0 * std_features0

    Fisher = (abs(mean_features0 - mean_features1)) / std_sum
    Fisher = np.array(Fisher)
    print('Fisher.shape', Fisher.shape)

    # sort the fisher from small to large

    feature_idx = np.arange(features.shape[1])
    Fisher_sorted = np.array(Fisher).argsort()
    # Fisher_sorted[0] has the smallest score
    #####################################################
    # now Fisher_sorted[::-1]]'s head is the index with the largest score!!
    sorted_feature_idx = feature_idx[Fisher_sorted[::-1]]
    return sorted_feature_idx
    #####################################################


def feature_selection(h, features, sorted_feature_idx):
    # only select h features
    print('sorted_feature_idx[:h]', sorted_feature_idx[:h])
    print('sorted_feature_idx', sorted_feature_idx.size)
    h_features = np.zeros((features.shape[0], h))
    for i in range(features.shape[0]):
        for j in range(h):
            h_features[i][j] = features[i][sorted_feature_idx[j]]
    return h_features

#########add these at line83######
# sorted_v_feature_idx=fisher(train_data,train_a_labels)
# train_v_data=feature_selection(150,train_data,sorted_v_feature_idx)

# sorted_a_feature_idx=fisher(train_data,train_a_labels)
# train_a_data=feature_selection(150,train_data,sorted_v_feature_idx)
