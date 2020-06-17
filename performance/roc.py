""" Evaluate binary classifier performance using ROC """
import numpy as np
import matplotlib.pyplot as plt


def detection_rate(output, target, thresh=0.5):
    """ Detection rate of a classifier for a given threshold """
    output = output.flatten()
    target = target.flatten()

    pred = output >= thresh
    assert pred.shape[0] == len(target)
    correct = np.sum((pred==1)*(target==1))
    n_one = np.sum(target==1)
    
    return correct / (1. if n_one == 0 else n_one)


def false_alarm_rate(output, target, thresh=0.5):
    """ False alarm rate of a classifier for a given threshold """
    output = output.flatten()
    target = target.flatten()

    pred = output >= thresh
    assert pred.shape[0] == len(target)
    correct = np.sum((pred==1)*(target==0))
    n_zero = np.sum(target==0)
    return correct / (1 if n_zero == 0 else n_zero)


def detection_atper(output, target, target_far=0.01, eps=1e-5, max_iter=100):
    """ Detection rate of a classifier at a given false alarm rate (binary search) """
    min_thresh = 0
    max_thresh = 1
    thresh = 0.5
    for _ in range(max_iter):
        far = false_alarm_rate(output, target, thresh=thresh)
        if far == target_far:
            break

        if far < target_far:
            # reduce threshold
            max_thresh = thresh
            thresh = 0.5 * (max_thresh + min_thresh)
        else:
            # increase threshold
            min_thresh = thresh
            thresh = 0.5 * (max_thresh + min_thresh)
        if abs(far - target_far) < eps:
            break

    return detection_rate(output, target, thresh=thresh)


def area_under_roc(output, target, num_steps=1000):
    """ Estimate area under ROC AUC (trapezoid integration) """
    auc = 0.
    prev_dr = 0.
    fars = np.linspace(0, 1., num_steps)
    for i, far in enumerate(fars):
        dr_t = detection_atper(output, target, target_far=far, eps=1e-5, max_iter=100)

        prev_far = fars[max(0, i-1)]
        delta = 0.5*(prev_dr + dr_t)*(far - prev_far)
        auc += delta
        prev_dr = dr_t

    return auc
