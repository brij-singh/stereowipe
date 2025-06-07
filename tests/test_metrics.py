import unittest
from biaswipe import metrics

class TestMetrics(unittest.TestCase):

    # Test methods for mock_stereotype_detector
    def test_mock_detector_stereotype_keyword(self):
        self.assertEqual(metrics.mock_stereotype_detector("This is a stereotype."), (True, 0.8))

    def test_mock_detector_stereotyped_keyword_case_insensitive(self):
        self.assertEqual(metrics.mock_stereotype_detector("This is STEREOTYPED."), (True, 0.8))

    def test_mock_detector_stereotypical_keyword(self):
        self.assertEqual(metrics.mock_stereotype_detector("This is stereotypical."), (True, 0.8))

    def test_mock_detector_stereotype_keyword_with_punctuation(self):
        self.assertEqual(metrics.mock_stereotype_detector("Is this a stereotype? Yes."), (True, 0.8))

    def test_mock_detector_no_keyword(self):
        self.assertEqual(metrics.mock_stereotype_detector("This is neutral."), (False, 0.0))

    def test_mock_detector_empty_string(self):
        self.assertEqual(metrics.mock_stereotype_detector(""), (False, 0.0))

    def test_mock_detector_keyword_substring_should_not_match(self):
        # "stereotypes" is not one of the keywords, nor "astereotype"
        self.assertEqual(metrics.mock_stereotype_detector("This concerns astereotype."), (False, 0.0))
        self.assertEqual(metrics.mock_stereotype_detector("This concerns stereotypes."), (False, 0.0))


    # Test methods for compute_stereotype_rate
    def test_csr_all_true(self):
        self.assertAlmostEqual(metrics.compute_stereotype_rate([True, True, True]), 1.0)

    def test_csr_all_false(self):
        self.assertAlmostEqual(metrics.compute_stereotype_rate([False, False]), 0.0)

    def test_csr_mixed(self):
        self.assertAlmostEqual(metrics.compute_stereotype_rate([True, False, True, False, False]), 0.4)

    def test_csr_empty_list(self):
        self.assertAlmostEqual(metrics.compute_stereotype_rate([]), 0.0)

    # Test methods for compute_stereotype_severity_score
    def test_csss_all_stereotyping(self):
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.5, 0.7, 0.9], [True, True, True]), 0.7)

    def test_csss_some_stereotyping(self):
        # (0.5 + 0.9 + 0.2) / 3 = 1.6 / 3 = 0.5333...
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.5, 0.7, 0.9, 0.2], [True, False, True, True]), 1.6 / 3)

    def test_csss_none_stereotyping(self):
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.5, 0.7], [False, False]), 0.0)

    def test_csss_empty_lists(self):
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([], []), 0.0)

    def test_csss_empty_scores_but_labels_true(self):
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([], [True, True]), 0.0)

    def test_csss_scores_but_empty_labels(self):
         self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.5, 0.6], []), 0.0)

    def test_csss_mismatched_lengths_ignored_extra_scores(self):
        # Only the first two scores (0.5, 0.6) are considered because labels is shorter.
        # (0.5 + 0.6) / 2 = 0.55
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.5, 0.6, 0.7], [True, True]), 0.55)

    def test_csss_mismatched_lengths_ignored_extra_labels(self):
        # Only the first two scores (0.5, 0.6) are considered because scores is shorter.
        # (0.5 + 0.6) / 2 = 0.55
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.5, 0.6], [True, True, False]), 0.55)

    def test_csss_one_item_true(self):
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.5, 0.6, 0.7], [False, True, False]), 0.6)

    def test_csss_one_item_false(self):
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.5], [False]), 0.0)

    def test_csss_one_item_true_single_element_lists(self):
        self.assertAlmostEqual(metrics.compute_stereotype_severity_score([0.9], [True]), 0.9)

if __name__ == '__main__':
    unittest.main()
