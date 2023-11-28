import numpy as np
import math

def countE(args):
    """
        count E
    """
    train_labels_address = './data/' + args.dataset + '/formatted_train_labels.npy'

    train_labels = np.load(train_labels_address, encoding='latin1')

    example_num = train_labels.shape[0]
    label_num = train_labels.shape[1]

    total_expected = 0

    for i in range(0, example_num):
        pos_per_exam = 0
        for j in range(0, label_num):
            if train_labels[i][j] == 1:
                pos_per_exam += 1
        if pos_per_exam != 0:
            total_expected += 1 / pos_per_exam
    result = total_expected / example_num

    return result


def count_true_(args):
    """
    get the true prior
    """
    train_labels_address = './data/' + args.dataset + '/formatted_train_labels.npy'

    train_labels = np.load(train_labels_address, encoding='latin1')

    example_num = train_labels.shape[0]
    label_num = train_labels.shape[1]

    result_E = countE(args)
    true_priors = []

    for i in range(0, label_num):
        total_pos = 0
        for j in range(0, example_num):
            if train_labels[j][i] == 1:
                total_pos += 1
        true_priors.append(total_pos / example_num)

    for i in range(0, label_num):
        true_priors[i] /= result_E

    return true_priors


def count_estimate_(args):
    """
    estimate the prior
    """
    train_sp_labels_address = './data/' + args.dataset + '/formatted_train_labels_obs.npy'

    train_sp_labels = np.load(train_sp_labels_address, encoding='latin1')
    
    example_num = train_sp_labels.shape[0]
    label_num = train_sp_labels.shape[1]

    result_E = countE(args)
    estimated_priors = []

    for i in range(0, label_num):
        total_pos = 0
        for j in range(0, example_num):
            if train_sp_labels[j][i] == 1:
                total_pos += 1
        estimated_priors.append(total_pos / example_num)

    for i in range(0, label_num):
        estimated_priors[i] /= result_E

    return estimated_priors