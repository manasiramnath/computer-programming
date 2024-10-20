
import random 

def main():
    pass

def split_training_testing(data, p_test):
    """ Split the data into training and testing sets. 
    
    Args:
    data (list): A list of data points.
    p_test (int): The percent of data to be used for testing (0-100).

    Returns:
    train (list): A list of data points for training.
    test (list): A list of data points for testing.

    The function randomly assigns data points to either the training or testing sets, 
    where the size of the sets is determined by the percent passed as an argument. 
    """
    # Check for invalid p_test values
    if p_test <= 0 or p_test >= 100:
        raise ValueError("p_test must be a positive integer or float in (0, 100) range.")
    
    # Check for insufficient data points
    if len(data) < 2:
        raise ValueError("There must be at least 2 data points.")
    
    # Check for invalid argument types
    if not isinstance(p_test, (int, float)):
        raise TypeError("p_test must be an integer or float.")

    if not isinstance(data, list):
        raise TypeError("data must be a list.")
    
    # Calculate the number of data points for testing
    num_testing = int(len(data) * (p_test / 100.0))

    # Shuffle data points
    random.shuffle(data)

    # Split data into training and testing sets
    train = data[num_testing:]
    test = data[:num_testing]

    return train, test

def confusion_matrix(predicted_values, actual_values, positive_class):
    """ Compute the confusion matrix.

    Parameters:
    - predicted_values (list): List of predicted values.
    - actual_values (list): List of actual values.
    - positive_class: Value indicating which class is positive.

    Returns:
    - TP (int): Number of true positives.
    - FP (int): Number of false positives.
    - TN (int): Number of true negatives.
    - FN (int): Number of false negatives.
    """
    unique_values = set(predicted_values)

    if len(unique_values) >= 2:
        if positive_class not in unique_values:
            raise ValueError("The positive class specified must be a value in predicted values.")
    
    # Initialise the counters
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    # Iterate over the predicted and actual values
    for predicted, actual in zip(predicted_values, actual_values):
        if predicted == actual == positive_class:
            TP += 1
        elif predicted == positive_class and actual != positive_class:
            FP += 1
        elif predicted != positive_class and actual == positive_class:
            FN += 1
        else:
            TN += 1

    return TP, FP, TN, FN

def accuracy(TP, FP, TN, FN):
    """ Estimate accuracy.

    Args:
    - TP (int): Number of true positives.
    - FP (int): Number of false positives.
    - TN (int): Number of true negatives.
    - FN (int): Number of false negatives.

    Returns:
    - float: Accuracy estimate.
    """
    # Check for negative values
    if TP < 0 or FP < 0 or TN < 0 or FN < 0:
        raise ValueError("One or more of the arguments are negative.")

    # Check for non-integer values
    if any(not isinstance(metric, int) for metric in (TP, FP, TN, FN)):
        raise TypeError("TP, FP, TN, FN must be integers.")
    
    denominator = TP + TN + FP + FN
    if denominator == 0:
        return float('nan')
    return (TP + TN) / denominator

def sensitivity(TP, FN):
    """ Estimate sensitivity (recall).

    Args:
    - TP (int): Number of true positives.
    - FN (int): Number of false negatives.

    Returns:
    - float: Sensitivity estimate.
    """
    # Check for negative values
    if TP < 0 or FN < 0:
        raise ValueError("One or more of the arguments are negative.")
    
    # Check for non-integer values
    if not isinstance(TP, int) or not isinstance(FN, int):
        raise TypeError("TP and FN must be integers.")    

    denominator = TP + FN
    if denominator == 0:
        return float('nan')
    return TP / denominator

def specificity(TN, FP):
    """ Estimate specificity (precision).

    Parameters:
    - TN (int): Number of true negatives.
    - FP (int): Number of false positives.

    Returns:
    - float: Specificity estimate.
    """
    # Check for negative values
    if TN < 0 or FP < 0:
        raise ValueError("One or more of the arguments are negative.")
    
    # Check for non-integer values
    if not isinstance(TN, int) or not isinstance(FP, int):
        raise TypeError("TN and FP must be integers.")
    
    denominator = TN + FP
    if denominator == 0:
        return float('nan')
    return TN / denominator

def pos_pred_val(TP, FP):
    """ Estimate positive predictive value.

    Args:
    - TP (int): Number of true positives.
    - FP (int): Number of false positives.

    Returns:
    - float: Positive predictive value estimate.
    """
    # Check for negative values
    if TP < 0 or FP < 0:
        raise ValueError("One or more of the arguments are negative.")
    
    # Check for non-integer values
    if not isinstance(TP, int) or not isinstance(FP, int):
        raise TypeError("TP and FP must be integers.")
    
    denominator = TP + FP
    if denominator == 0:
        return float('nan')
    return TP / denominator

def neg_pred_val(TN, FN):
    """ Estimate negative predictive value.

    Parameters:
    - TN (int): Number of true negatives.
    - FN (int): Number of false negatives.

    Returns:
    - float: Negative predictive value estimate.
    """
    # Check for negative values
    if TN < 0 or FN < 0:
        raise ValueError("One or more of the arguments are negative.")
    
    # Check for non-integer values
    if not isinstance(TN, int) or not isinstance(FN, int):
        raise TypeError("TN and FN must be integers.")
    
    denominator = TN + FN
    if denominator == 0:
        return float('nan')
    return TN / denominator

def print_eval_metrics(predicted_values, actual_values, positive_class):
    """ Print evaluation metrics.

    Args:
    - predicted_values (list): List of predicted values.
    - actual_values (list): List of actual values.
    - positive_class: Value indicating which class is positive.
    """

    # Calculate confusion matrix metrics
    TP, FP, TN, FN = confusion_matrix(predicted_values, actual_values, positive_class)

    # Calculate evaluation metrics
    acc = accuracy(TP, FP, TN, FN)
    sens = sensitivity(TP, FN)
    spec = specificity(TN, FP)
    pos_pred = pos_pred_val(TP, FP)
    neg_pred = neg_pred_val(TN, FN)

    # Print the evaluation metrics
    print(f"Accuracy: {acc}")
    print(f"Sensitivity: {sens}")
    print(f"Specificity: {spec}")
    print(f"Positive predictive value: {pos_pred}")
    print(f"Negative predictive value: {neg_pred}")