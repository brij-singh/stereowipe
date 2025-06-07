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

    # Test methods for compute_csss
    def setUp_csss_data(self):
        self.sample_prompts = {
            "p1": {"text": "t1", "category": "catA"},
            "p2": {"text": "t2", "category": "catB"},
            "p3": {"text": "t3", "category": "catA"},
            "p4": {"text": "t4", "category": "catC"}
        }
        self.prompt_ids = ["p1", "p2", "p3", "p4"]

    def test_csss_basic(self):
        self.setUp_csss_data()
        labels = [True, True, True, False]
        scores = [0.8, 0.6, 0.4, 0.9]
        expected = {"catA": (0.8 + 0.4) / 2, "catB": 0.6} # catC has no stereotyping response

        # Suppress print warnings for this test if any prompt issues were to occur, though not expected here
        original_print = metrics.print
        metrics.print = lambda *args, **kwargs: None
        result = metrics.compute_csss(self.sample_prompts, labels, scores, self.prompt_ids)
        metrics.print = original_print

        self.assertEqual(len(result), len(expected))
        for category, avg_score in expected.items():
            self.assertIn(category, result)
            self.assertAlmostEqual(result[category], avg_score)

    def test_csss_no_stereotyping(self):
        self.setUp_csss_data()
        labels = [False, False, False, False]
        scores = [0.8, 0.6, 0.4, 0.9]
        expected = {} # No categories will have stereotyping responses

        original_print = metrics.print
        metrics.print = lambda *args, **kwargs: None
        result = metrics.compute_csss(self.sample_prompts, labels, scores, self.prompt_ids)
        metrics.print = original_print

        self.assertEqual(result, expected)

    def test_csss_all_stereotyping_one_category(self):
        sample_prompts_one_cat = {"p1": {"text": "t1", "category": "catA"}, "p2": {"text": "t2", "category": "catA"}}
        prompt_ids_one_cat = ["p1", "p2"]
        labels = [True, True]
        scores = [0.8, 0.4]
        expected = {"catA": 0.6}

        original_print = metrics.print
        metrics.print = lambda *args, **kwargs: None
        result = metrics.compute_csss(sample_prompts_one_cat, labels, scores, prompt_ids_one_cat)
        metrics.print = original_print

        self.assertEqual(len(result), len(expected))
        for category, avg_score in expected.items():
            self.assertIn(category, result)
            self.assertAlmostEqual(result[category], avg_score)

    def test_csss_empty_inputs(self):
        expected = {}
        original_print = metrics.print # Assuming compute_csss might print for mismatched lengths
        metrics.print = lambda *args, **kwargs: None
        result = metrics.compute_csss({}, [], [], [])
        metrics.print = original_print
        self.assertEqual(result, expected)

    def test_csss_mismatched_input_lengths(self):
        self.setUp_csss_data()
        # Expect error print and empty dict
        original_print = metrics.print
        captured_errors = []
        metrics.print = lambda *args, **kwargs: captured_errors.append(args[0])

        result = metrics.compute_csss(self.sample_prompts, [True], [0.5, 0.6], ["p1"])
        metrics.print = original_print

        self.assertEqual(result, {})
        self.assertTrue(any("Input lists (stereotype_labels, severity_scores, prompt_ids) must have the same length." in error for error in captured_errors))

    def test_csss_prompt_id_not_in_prompts(self):
        self.setUp_csss_data()
        labels = [True]
        scores = [0.8]
        prompt_ids_invalid = ["p_non_existent"]
        expected = {}

        original_print = metrics.print
        captured_warnings = []
        metrics.print = lambda *args, **kwargs: captured_warnings.append(args[0])

        result = metrics.compute_csss(self.sample_prompts, labels, scores, prompt_ids_invalid)
        metrics.print = original_print

        self.assertEqual(result, expected)
        self.assertTrue(any("Prompt ID 'p_non_existent' not found" in warning for warning in captured_warnings))

    def test_csss_prompt_missing_category_key(self):
        sample_prompts_no_cat_key = {"p1": {"text": "t1"}} # No "category" key
        labels = [True]
        scores = [0.8]
        prompt_ids_no_cat_key = ["p1"]
        expected = {}

        original_print = metrics.print
        captured_warnings = []
        metrics.print = lambda *args, **kwargs: captured_warnings.append(args[0])

        result = metrics.compute_csss(sample_prompts_no_cat_key, labels, scores, prompt_ids_no_cat_key)
        metrics.print = original_print

        self.assertEqual(result, expected)
        self.assertTrue(any("Prompt ID 'p1' not found in prompts data or 'category' key missing." in warning for warning in captured_warnings))

    def test_csss_category_not_string(self):
        self.setUp_csss_data()
        prompts_cat_not_string = {"p1": {"text": "t1", "category": 123}} # Category is int
        labels = [True]
        scores = [0.8]
        prompt_ids_cat_not_string = ["p1"]
        expected = {}

        original_print = metrics.print
        captured_warnings = []
        metrics.print = lambda *args, **kwargs: captured_warnings.append(args[0])

        result = metrics.compute_csss(prompts_cat_not_string, labels, scores, prompt_ids_cat_not_string)
        metrics.print = original_print

        self.assertEqual(result, expected)
        self.assertTrue(any("Category for prompt_id 'p1' is not a string." in warning for warning in captured_warnings))

    # Test methods for compute_wosi
    def test_wosi_basic(self):
        csss = {"catA": 0.8, "catB": 0.6}
        weights = {"catA": 0.7, "catB": 0.3} # Weights sum to 1
        # Expected: (0.8 * 0.7) + (0.6 * 0.3) = 0.56 + 0.18 = 0.74
        # Denominator: 0.7 + 0.3 = 1.0
        # WOSI = 0.74 / 1.0 = 0.74
        original_print = metrics.print
        metrics.print = lambda *args, **kwargs: None # Suppress potential warnings if any
        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.74)
        metrics.print = original_print

    def test_wosi_weights_do_not_sum_to_one(self):
        csss = {"catA": 0.8, "catB": 0.6}
        weights = {"catA": 2.0, "catB": 1.0}
        # Expected: ((0.8 * 2.0) + (0.6 * 1.0)) / (2.0 + 1.0)
        # = (1.6 + 0.6) / 3.0 = 2.2 / 3.0
        original_print = metrics.print
        metrics.print = lambda *args, **kwargs: None
        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 2.2 / 3.0)
        metrics.print = original_print

    def test_wosi_category_in_csss_not_in_weights(self):
        csss = {"catA": 0.8, "catB": 0.6, "catC": 0.9} # catC not in weights
        weights = {"catA": 0.7, "catB": 0.3}
        # Expected: ((0.8 * 0.7) + (0.6 * 0.3)) / (0.7 + 0.3) = 0.74 / 1.0 = 0.74
        # catC is ignored.
        original_print = metrics.print
        captured_warnings = []
        metrics.print = lambda *args, **kwargs: captured_warnings.append(args[0])

        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.74)
        self.assertTrue(any("Category 'catC' found in CSSS scores but not in category weights." in w for w in captured_warnings))
        metrics.print = original_print

    def test_wosi_category_in_weights_not_in_csss(self):
        csss = {"catA": 0.8} # catB from weights is not here
        weights = {"catA": 0.7, "catB": 0.3}
        # Expected: (0.8 * 0.7) / 0.7 = 0.8
        original_print = metrics.print
        metrics.print = lambda *args, **kwargs: None
        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.8)
        metrics.print = original_print

    def test_wosi_empty_csss(self):
        csss = {}
        weights = {"catA": 0.7, "catB": 0.3}
        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.0)

    def test_wosi_empty_weights(self):
        csss = {"catA": 0.8, "catB": 0.6}
        weights = {}

        original_print = metrics.print
        captured_warnings = []
        metrics.print = lambda *args, **kwargs: captured_warnings.append(args[0])

        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.0)
        self.assertTrue(any("Category weights are empty" in w for w in captured_warnings))
        metrics.print = original_print

    def test_wosi_no_matching_categories(self):
        csss = {"catA": 0.8, "catB": 0.6}
        weights = {"catC": 0.7, "catD": 0.3} # No common categories

        original_print = metrics.print
        captured_warnings = []
        metrics.print = lambda *args, **kwargs: captured_warnings.append(args[0])

        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.0)
        # Expect two warnings for catA and catB not in weights, and one for sum_of_weights_used being 0
        self.assertTrue(any("Category 'catA' found in CSSS scores but not in category weights." in w for w in captured_warnings))
        self.assertTrue(any("Category 'catB' found in CSSS scores but not in category weights." in w for w in captured_warnings))
        self.assertTrue(any("Sum of weights used for WOSI calculation is 0.0" in w for w in captured_warnings))
        metrics.print = original_print

    def test_wosi_invalid_weight_type(self):
        csss = {"catA": 0.8, "catB": 0.5, "catC": 0.9}
        weights = {"catA": "invalid_weight", "catB": 0.5, "catC": -0.1} # catA invalid, catC negative
        # Expected: (0.5 * 0.5) / 0.5 = 0.5
        # catA skipped due to invalid weight type, catC skipped due to negative weight

        original_print = metrics.print
        captured_warnings = []
        metrics.print = lambda *args, **kwargs: captured_warnings.append(args[0])

        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.5)
        self.assertTrue(any("Weight for category 'catA' is not a number" in w for w in captured_warnings))
        self.assertTrue(any("Weight for category 'catC' is negative" in w for w in captured_warnings))
        metrics.print = original_print

    def test_wosi_all_weights_zero_or_invalid(self):
        csss = {"catA": 0.8, "catB": 0.6}
        weights = {"catA": 0.0, "catB": "invalid"}

        original_print = metrics.print
        captured_warnings = []
        metrics.print = lambda *args, **kwargs: captured_warnings.append(args[0])

        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.0)
        self.assertTrue(any("Weight for category 'catB' is not a number" in w for w in captured_warnings))
        self.assertTrue(any("Sum of weights used for WOSI calculation is 0.0" in w for w in captured_warnings))
        metrics.print = original_print

    def test_wosi_single_category_match_zero_weight(self):
        csss = {"catA": 0.8, "catB": 0.5}
        weights = {"catA": 0.0, "catC": 0.5} # catA has zero weight, catB not in weights

        original_print = metrics.print
        captured_warnings = []
        metrics.print = lambda *args, **kwargs: captured_warnings.append(args[0])

        self.assertAlmostEqual(metrics.compute_wosi(csss, weights), 0.0)
        self.assertTrue(any("Category 'catB' found in CSSS scores but not in category weights" in w for w in captured_warnings))
        self.assertTrue(any("Sum of weights used for WOSI calculation is 0.0" in w for w in captured_warnings))
        metrics.print = original_print


if __name__ == '__main__':
    unittest.main()
