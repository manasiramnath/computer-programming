import unittest
import math
import class_eval
import random

class TestSplit(unittest.TestCase):
    """Tests for split_training_testing function"""

    def test_split_list(self):
        """Test for basic list to check if lengths are equal and elements are unchanged."""
        inputted_data = [1, 2, 3, 4]
        p_test = 50
        train, test = class_eval.split_training_testing(inputted_data, p_test)
        expected_output = (len(inputted_data) - len(test), len(test))
        self.assertEqual((len(train), len(test)), expected_output, "Lengths match.")
        self.assertEqual(set(train + test), set(inputted_data), "Matching elements.")
    
    def test_split_invalidperc_value(self):
        """Test for invalid p_test value."""
        inputted_data = [1, 2, 3, 4]
        p_test = -10
        with self.assertRaises(ValueError):
            class_eval.split_training_testing(inputted_data, p_test)
    
    def test_split_invalidperc_type(self):
        """Test for invalid p_test type."""
        inputted_data = [1, 2, 3, 4]
        p_test = 'a'
        with self.assertRaises(TypeError):
            class_eval.split_training_testing(inputted_data, p_test)
    
    def test_split_perc_float(self):
        """Test for float p_test."""
        inputted_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        p_test = 27.3
        train, test = class_eval.split_training_testing(inputted_data, p_test)
        expected_output = (len(train), len(test))
        self.assertEqual((len(train), len(test)), expected_output, "Length matches with float percentage.")
    
    def test_split_no_training(self):
        """Test for no training data (p_test = 100)."""
        inputted_data = [1, 2, 3, 4]
        p_test = 100
        with self.assertRaises(ValueError):
            class_eval.split_training_testing(inputted_data, p_test)

    def test_split_no_testing(self):
        """Test for no testing data (p_test = 0)."""
        inputted_data = [1, 2, 3, 4]
        p_test = 0
        with self.assertRaises(ValueError):
            class_eval.split_training_testing(inputted_data, p_test)

    def test_split_invaliddata_type(self):
        """Test for invalid data type."""
        inputted_data = 1
        p_test = 50
        with self.assertRaises(TypeError):
            class_eval.split_training_testing(inputted_data, p_test)
    
    def test_split_insufficientdata(self):
        """Test for insufficient data."""
        inputted_data = [1]
        p_test = 50
        with self.assertRaises(ValueError):
            class_eval.split_training_testing(inputted_data, p_test)

    def test_split_mixed_list(self):
        """Test for mixed inputted list, which includes non-integer items."""
        inputted_data = [1, 2, '?', 4]
        p_test = 50
        train, test = class_eval.split_training_testing(inputted_data, p_test)
        self.assertEqual(len(train) + len(test), len(inputted_data), "Lengths match.")
        self.assertEqual(set(train + test), set(inputted_data), "Matching elements.")
    
    def test_split_list_randomness(self):
        """Test for randomness of list."""
        inputted_data = [1, 2, 3, 4]
        p_test = 50
        unique_sets = set()
        for _ in range(10):
            _, test = class_eval.split_training_testing(inputted_data, p_test)
            unique_sets.add(tuple(test))
        self.assertTrue(len(unique_sets) > 1, "Randomness of list.")
    
    def test_split_nested_list(self):
        """Test for nested lists."""
        inputted_data = [[1, 2], [3, 4], [5, 6]]
        p_test = 50
        train, test = class_eval.split_training_testing(inputted_data, p_test)
        # Convert to tuples to compare
        train = tuple(tuple(x) for x in train)
        test = tuple(tuple(x) for x in test)
        self.assertEqual(len(train) + len(test), len(inputted_data), "Lengths match.")
        self.assertEqual(set(tuple(x) for x in train + test), set(tuple(x) for x in inputted_data), "Matching elements.")

class TestConfusionMatrix(unittest.TestCase):
    """Tests for confusion_matrix function"""
    def test_confmatrix_values(self):
        """Test for basic values."""
        predicted_values = [0, 1, 1, 0]
        actual_values = [1, 1, 0, 0]
        positive_class = 1
        TP, FP, TN, FN = class_eval.confusion_matrix(predicted_values, actual_values, positive_class)
        self.assertEqual(TP, 1, "True positives.")
        self.assertEqual(FP, 1, "False positives.")
        self.assertEqual(TN, 1, "True negatives.")
        self.assertEqual(FN, 1, "False negatives.")
    
    def test_confmatrix_positive_class_invalid(self):
        """Test for invalid positive_class value."""
        predicted_values = [0, 1, 1, 0]
        actual_values = [1, 1, 0, 0]
        positive_class = 10
        with self.assertRaises(ValueError):
            class_eval.confusion_matrix(predicted_values, actual_values, positive_class)

    def test_confmatrix_empty_list(self):
        """Test for empty predicted values"""
        predicted_values = []
        actual_values = []
        positive_class = 1
        expected_output = (0, 0, 0, 0)
        self.assertEqual(class_eval.confusion_matrix(predicted_values, actual_values, positive_class),
                          expected_output, "Empty list.")
    
    def test_confmatrix_no_negative(self):
        """Test for no negative values."""
        predicted_values = [1, 1, 1, 1]
        actual_values = [1, 1, 1, 1]
        positive_class = 1
        TP, FP, TN, FN = class_eval.confusion_matrix(predicted_values, actual_values, positive_class)
        self.assertEqual(TP, 4, "True positives.")
        self.assertEqual(FP, 0, "False positives.")
        self.assertEqual(TN, 0, "True negatives.")
        self.assertEqual(FN, 0, "False negatives.")
    
    def test_confmatrix_no_positive(self):
        """Test for no positive values."""
        predicted_values = [0, 0, 0, 0]
        actual_values = [0, 0, 0, 0]
        positive_class = 1
        TP, FP, TN, FN = class_eval.confusion_matrix(predicted_values, actual_values, positive_class)
        self.assertEqual(TP, 0, "True positives.")
        self.assertEqual(FP, 0, "False positives.")
        self.assertEqual(TN, 4, "True negatives.")
        self.assertEqual(FN, 0, "False negatives.")

    def test_confmatrix_mixed_lists(self):
        """Test for mixed lists, which includes non-integer items."""
        predicted_values = [1, 1, 'x', 0]
        actual_values = [1, 'x', 0, 1]
        positive_class = 1
        TP, FP, TN, FN = class_eval.confusion_matrix(predicted_values, actual_values, positive_class)
        self.assertEqual(TP, 1, "True positives.")
        self.assertEqual(FP, 1, "False positives.")
        self.assertEqual(TN, 1, "True negatives.")
        self.assertEqual(FN, 1, "False negatives.")

class TestAccuracy(unittest.TestCase):
    """Tests for accuracy function"""
    def test_accuracy_values(self):
        """Test for basic values."""
        TP = 1
        FP = 1
        TN = 1
        FN = 1
        self.assertEqual(class_eval.accuracy(TP, FP, TN, FN), 0.5, "Accuracy value.")
    
    def test_accuracy_zeros(self):
        """Tests when denominator is zero."""
        result = class_eval.accuracy(0, 0, 0, 0)
        self.assertTrue(math.isnan(result), "All zeros, accuracy should be NaN.")
    
    def test_accuracy_neg(self):
        """Tests when any value is negative."""
        TP = -1
        FP = 1
        TN = 1
        FN = 1
        with self.assertRaises(ValueError):
            class_eval.accuracy(TP, FP, TN, FN)
    
    def test_accuracy_nonint(self):
        """Tests when any value is non-integer."""
        TP = 'b'
        FP = 1
        TN = 1
        FN = 1
        with self.assertRaises(TypeError):
            class_eval.accuracy(TP, FP, TN, FN)

class TestSensitivity(unittest.TestCase):
    """Tests for sensitivity function"""
    def test_sens_values(self):
        """Test for basic values."""
        TP = 1
        FN = 1
        self.assertEqual(class_eval.sensitivity(TP, FN), 0.5, "Sensitivity value.")
    
    def test_sens_zeros(self):
        """Tests when denominator is zero."""
        result = class_eval.sensitivity(0,0)
        self.assertTrue(math.isnan(result), "All zeros, sensitivity should be NaN.")
    
    def test_sens_neg(self):
        """Tests when any value is negative."""
        TP = -1
        FN = 1
        with self.assertRaises(ValueError):
            class_eval.sensitivity(TP, FN)
    
    def test_sens_nonint(self):
        """Tests when any value is non-integer."""
        TP = 'b'
        FN = 1
        with self.assertRaises(TypeError):
            class_eval.sensitivity(TP, FN)

class TestSpecificity(unittest.TestCase):
    """Tests for specificity function"""
    def test_spec_values(self):
        """Test for basic values."""
        TN = 1
        FP = 1
        self.assertEqual(class_eval.specificity(TN, FP), 0.5, "Specificity value.")
    
    def test_spec_zeros(self):
        """Tests when denominator is zero."""
        result = class_eval.specificity(0,0)
        self.assertTrue(math.isnan(result), "All zeros, specificity should be NaN.")
    
    def test_spec_neg(self):
        """Tests when any value is negative."""
        TN = -1
        FP = 1
        with self.assertRaises(ValueError):
            class_eval.specificity(TN, FP)
    
    def test_spec_nonint(self):
        """Tests when any value is non-integer."""
        TN = 'b'
        FP = 1
        with self.assertRaises(TypeError):
            class_eval.specificity(TN, FP)

class TestPosPred(unittest.TestCase):
    """Tests for positive predictive value function"""
    def test_pospred_values(self):
        """Test for basic values."""
        TP = 1
        FP = 1
        self.assertEqual(class_eval.pos_pred_val(TP, FP), 0.5, "Positive predictive value.")
    
    def test_pospred_zeros(self):
        """Tests when denominator is zero."""
        result = class_eval.pos_pred_val(0, 0)
        self.assertTrue(math.isnan(result), "All zeros, positive predictive value should be NaN.")
    
    def test_pospred_neg(self):
        """Tests when any value is negative."""
        TP = -1
        FP = 1
        with self.assertRaises(ValueError):
            class_eval.pos_pred_val(TP, FP)
    
    def test_pospred_nonint(self):
        """Tests when any value is non-integer."""
        TP = 'b'
        FP = 1
        with self.assertRaises(TypeError):
            class_eval.pos_pred_val(TP, FP)

class TestNegPred(unittest.TestCase):
    """Tests for positive predictive value function"""
    def test_negpred_values(self):
        """Test for basic values."""
        TN = 1
        FN = 1
        self.assertEqual(class_eval.neg_pred_val(TN, FN), 0.5, "Negative predictive value.")
    
    def test_negpred_zeros(self):
        """Tests when denominator is zero."""
        result = class_eval.neg_pred_val(0, 0)
        self.assertTrue(math.isnan(result), "All zeros, negative predictive value should be NaN.")
    
    def test_negpred_neg(self):
        """Tests when any value is negative."""
        TN = -1
        FN = 1
        with self.assertRaises(ValueError):
            class_eval.neg_pred_val(TN, FN)
    
    def test_negpred_nonint(self):
        """Tests when any value is non-integer."""
        TN = 'b'
        FN = 1
        with self.assertRaises(TypeError):
            class_eval.neg_pred_val(TN, FN)
    

if __name__ == '__main__':
    unittest.main()