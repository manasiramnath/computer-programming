import random

def main():
    pass

def split_training_testing(data, p_test):
    """Assume data is a list of length of at least 10
    and p_test is a positive number less than 100.
    Take a list of data points and split it randomly into a training set
    and a testing set, where testing set is p_test percent of the data.
    Return the training and testing sets, each as a list of data points.
    """
    assert (p_test > 0 and p_test < 100), 'Percent for testing set should be a positive number smaller than 100.'
    l = len(data)
    assert l >= 10, 'Data are too small to split.'
    test_indeces = random.sample(range(l), int(round(l*p_test/100, 0)))
    train_set = [data[i] for i in range(l) if i not in test_indeces]
    test_set = [data[i] for i in test_indeces]
    return train_set, test_set

def confusion_matrix(vals_predicted, vals_actual, pos_val):
    """Take a non-empty list of predicted labels, a non-empty list of
    actual labels, and the value for the positive label and return the number
    of true positives, false positives, true negatives, and false negatives.
    """
    assert (len(vals_predicted) != 0 and len(vals_actual) != 0), 'Predicted and actual labels should not be empty.'
    assert len(vals_predicted) == len(vals_actual), 'Predicted and actual labels are not the same length.'
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(vals_predicted)):
        pred = vals_predicted[i]
        if pred == pos_val:
            if vals_actual[i] == pred:
                tp += 1
            else:
                fp += 1
        else:
            if vals_actual[i] == pred:
                tn += 1
            else:
                fn += 1
    return tp, fp, tn, fn

def accuracy(tp, fp, tn, fn):
    """Assume tp, fp, tn, and fn are non-negative integers.
    Return the accuracy, which is (tp + tn) / (tp + fp + tn + fn).
    """
    assert (tp >= 0 and fp >= 0 and tn >= 0 and fn >= 0), 'Function expects non-negative integers as arguments.'
    try:
        return (tp + tn) / (tp + fp + tn + fn)
    except ZeroDivisionError:
        return float('nan')

def sensitivity(tp, fn):
    """Assume tp and fn are non-negative integers.
    Return the sensitivity, which is tp / (tp + fn).
    """
    assert (tp >= 0 and fn >= 0), 'Function expects non-negative integers as arguments.'
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return float('nan')

def specificity(tn, fp):
    """Assume tn and fp are non-negative integers.
    Return the specificity, which is tn / (tn + fp).
    """
    assert (tn >= 0 and fp >= 0), 'Function expects non-negative integers as arguments.'
    try:
        return tn / (tn + fp)
    except ZeroDivisionError:
        return float('nan')

def pos_pred_val(tp, fp):
    """Assume tp and fp are non-negative integers.
    Return the positive predictive value, which is tp / (tp + fp).
    """
    assert (tp >= 0 and fp >= 0), 'Function expects non-negative integers as arguments.'
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return float('nan')

def neg_pred_val(tn, fn):
    """Assume tn and fn are non-negative integers.
    Return the negative predictive value, which is tn / (tn + fn).
    """
    assert (tn >= 0 and fn >= 0), 'Function expects non-negative integers as arguments.'
    try:
        return tn / (tn + fn)
    except ZeroDivisionError:
        return float('nan')

def print_eval_metrics(vals_predicted, vals_actual, pos_val):
    """Take a list of predicted labels, a list of actual labels, and the value
    for the positive label and print the accuracy, sensitivty, specificity,
    positive predictive value, and negative predictive value.
    """
    tp, fp, tn, fn = confusion_matrix(vals_predicted, vals_actual, pos_val)
    print('Accuracy:', accuracy(tp, fp, tn, fn))
    print('Sensitivity:', sensitivity(tp, fn))
    print('Specificity:', specificity(tn, fp))
    print('Positive predictive value:', pos_pred_val(tp, fp))
    print('Negative predictive value:', neg_pred_val(tn, fn))

if __name__ == '__main__':
    main()
