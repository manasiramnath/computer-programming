import unittest
from leading import *

class TestLeading(unittest.TestCase):
    """Tests for leading_substrings."""

    def test_empty(self):
        """Test the empty string."""
        output = leading_substrings('')
        expected = []
        self.assertEqual(expected, output, 'Argument is empty string.')

    def test_single_letter(self):
        """Test a one-character string."""
        output = leading_substrings('x')
        expected = ['x']
        self.assertEqual(expected, output, 'Argument is a single letter.')

    def test_word(self):
        """Test a longer word."""
        output = leading_substrings('water')
        expected = ['w', 'wa', 'wat', 'wate', 'water']
        self.assertEqual(expected, output, 'Argument is a longer word.')

if __name__ == '__main__':
    unittest.main()
