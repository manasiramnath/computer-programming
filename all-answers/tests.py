import unittest
from class_eval import *
import math

class TestSplit(unittest.TestCase):
    """Tests for split_training_testing."""

    def test_split_empty(self):
        """Test a split where the percent for the testing set
        results in an empty testing set."""
        data = [i for i in range(10)]
        expected = [len(data), 0]
        result = split_training_testing(data, 2)
        self.assertEqual(expected, [len(result[0]), len(result[1])], "Split results in an empty testing set.")

    def test_split_full(self):
        """Test a split where the percent for the testing set
        results in an empty training set."""
        data = [i for i in range(10)]
        expected = [0, len(data)]
        result = split_training_testing(data, 98)
        self.assertEqual(expected, [len(result[0]), len(result[1])], "Split results in an empty training set.")

    def test_split_20(self):
        """Test a split where the percent for the testing set is 20%."""
        data = [i for i in range(10)]
        expected = [8, 2]
        result = split_training_testing(data, 20)
        self.assertEqual(expected, [len(result[0]), len(result[1])], "Split results for a 20/80 split.")

class TestConfusionMatrix(unittest.TestCase):
    """Tests for confusion_matrix."""

    def test_confusion_perfect_pos(self):
        """Test confusion matrix for perfectly predicted all positive data."""
        actual = [1, 1, 1]
        pred = [1, 1, 1]
        expected = (3, 0, 0, 0)
        result = confusion_matrix(pred, actual, 1)
        self.assertEqual(expected, result, "Confusion matrix for perfectly predicted all positive data.")

    def test_confusion_perfect_neg(self):
        """Test confusion matrix for perfectly predicted all negative data."""
        actual = [0, 0, 0]
        pred = [0, 0, 0]
        expected = (0, 0, 3, 0)
        result = confusion_matrix(pred, actual, 1)
        self.assertEqual(expected, result, "Confusion matrix for perfectly predicted all negative data.")

    def test_confusion_perfect_mix(self):
        """Test confusion matrix for perfectly predicted mixed data."""
        actual = [1, 1, 0]
        pred = [1, 1, 0]
        expected = (2, 0, 1, 0)
        result = confusion_matrix(pred, actual, 1)
        self.assertEqual(expected, result, "Confusion matrix for perfectly predicted mixed data.")

    def test_confusion_worst_mix(self):
        """Test confusion matrix for worst-case predicted mixed data."""
        actual = [1, 1, 0]
        pred = [0, 0, 1]
        expected = (0, 1, 0, 2)
        result = confusion_matrix(pred, actual, 1)
        self.assertEqual(expected, result, "Confusion matrix for worst-case predicted mixed data.")

    def test_confusion_label_type(self):
        """Test confusion matrix for labels of different data types."""
        actual = ['a', 'a', False]
        pred = ['a', False, 'a']
        expected = (1, 1, 0, 1)
        result = confusion_matrix(pred, actual, 'a')
        self.assertEqual(expected, result, "Confusion matrix for labels of different data types.")

class TestAccuracy(unittest.TestCase):
    """Tests for accuracy."""

    def test_accuracy_zerodenom(self):
        """Test accuracy for confusion matrix of zeros."""
        tp, fp, tn, fn = 0, 0, 0, 0
        result = accuracy(tp, fp, tn, fn)
        self.assertTrue(math.isnan(result), "Accuracy for confusion matrix of zeros.")

    def test_accuracy_regular(self):
        """Test accuracy for common values."""
        tp, fp, tn, fn = 2, 2, 5, 0
        expected = 7/9
        result = accuracy(tp, fp, tn, fn)
        self.assertAlmostEqual(expected, result, places = 5,
             msg = "Accuracy for common values.")

class TestSensitivity(unittest.TestCase):
    """Tests for sensitivity."""

    def test_sensitivity_zerodenom(self):
        """Test sensitivity for zero denominator."""
        a, b = 0, 0
        result = sensitivity(a, b)
        self.assertTrue(math.isnan(result), "Sensitivity for zero denominator.")

    def test_sensitivity_regular(self):
        """Test sensitivity for common values."""
        a, b = 3, 5
        expected = 3/8
        result = sensitivity(a, b)
        self.assertAlmostEqual(expected, result, places = 5,
             msg = "Sensitivity for common values.")

class TestSpecificity(unittest.TestCase):
    """Tests for specificity."""

    def test_specificity_zerodenom(self):
        """Test specificity for zero denominator."""
        a, b = 0, 0
        result = specificity(a, b)
        self.assertTrue(math.isnan(result), "Specificity for zero denominator.")

    def test_specificity_regular(self):
        """Test specificity for common values."""
        a, b = 4, 1
        expected = 4/5
        result = specificity(a, b)
        self.assertAlmostEqual(expected, result, places = 5,
             msg = "Specificity for common values.")

class TestPosPredVal(unittest.TestCase):
    """Tests for positive predictive value."""

    def test_pos_pred_val_zerodenom(self):
        """Test positive predictive value for zero denominator."""
        a, b = 0, 0
        result = pos_pred_val(a, b)
        self.assertTrue(math.isnan(result), "Positive predictive value for zero denominator.")

    def test_pos_pred_val_regular(self):
        """Test positive predictive value for common values."""
        a, b = 3, 7
        expected = 3/10
        result = pos_pred_val(a, b)
        self.assertAlmostEqual(expected, result, places = 5,
             msg = "Positive predictive value for common values.")

class TestNegPredVal(unittest.TestCase):
    """Tests for negative predictive value."""

    def test_neg_pred_val_zerodenom(self):
        """Test negative predictive value for zero denominator."""
        a, b = 0, 0
        result = neg_pred_val(a, b)
        self.assertTrue(math.isnan(result), "Negative predictive value for zero denominator.")

    def test_neg_pred_val_regular(self):
        """Test negative predictive value for common values."""
        a, b = 0, 2
        expected = 0/2
        result = neg_pred_val(a, b)
        self.assertAlmostEqual(expected, result, places = 5,
             msg = "Negative predictive value for common values.")


if __name__ == '__main__':
    unittest.main()
