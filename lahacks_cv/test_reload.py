"""Regression checks for the actual remote failure; no training or network."""
import unittest
import numpy as np
from .metrics import verify_rf_reload


class RFReloadTests(unittest.TestCase):
    def test_parallel_roundoff_passes_with_identical_decisions(self):
        original = np.array([.24132, .122145, .6, .9])
        restored = np.nextafter(original, np.ones_like(original))
        result = verify_rf_reload(original, restored)
        self.assertTrue(result['threshold_decisions_identical'])
        self.assertLess(result['maximum_absolute_difference'], 1e-15)

    def test_material_probability_change_fails(self):
        with self.assertRaises(AssertionError):
            verify_rf_reload([.2, .8], [.20001, .8])

    def test_roundoff_crossing_threshold_still_fails(self):
        with self.assertRaises(AssertionError):
            verify_rf_reload([np.nextafter(.5, 0)], [.5])

    def test_nonfinite_or_misaligned_probabilities_fail(self):
        for a, b in (([np.nan], [np.nan]), ([.2], [.2, .3]), ([.2], [np.inf])):
            with self.assertRaises(ValueError):
                verify_rf_reload(a, b)


if __name__ == '__main__':
    unittest.main(verbosity=2)
