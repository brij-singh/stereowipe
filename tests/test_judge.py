import unittest
import os
import json
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Assuming biaswipe is installed or PYTHONPATH is set up correctly
from biaswipe.judge import (
    Judge,
    MockJudge,
    OpenAIJudge, # Will be used to test MissingApiKeyError
    AnthropicJudge, # Will be used to test MissingApiKeyError
    get_consensus,
    run_judge_ensemble,
    _generate_cache_key, # For specific cache key tests if needed
    JUDGE_PROMPT_PATH, # This is "judge_prompt.txt"
    CACHE_DIR_BASE, # This is Path(".cache/judgments/")
    MissingApiKeyError,
    ApiCallError,
    get_judged_response_with_cache # Test this directly for more focused cache tests
)

# --- Test Configuration ---
TEST_JUDGE_PROMPT_CONTENT = "Test Judge Prompt: {response_text}"
# Define a separate cache directory for testing to avoid conflicts
TEST_CACHE_DIR = Path(".test_cache_judge/judgments/")

# --- Helper Failing Judge ---
class FailingJudge(Judge):
    def __init__(self, name="FailingJudge"):
        self.name = name

    def judge_response(self, response_text: str) -> dict:
        raise ApiCallError(f"{self.name}: Simulated failure during judgment.")

class TestJudgeSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a dummy judge_prompt.txt in the project root
        # as the main code expects it there.
        cls.root_dir = Path(__file__).parent.parent # stereowipe/
        cls.actual_judge_prompt_file = cls.root_dir / JUDGE_PROMPT_PATH

        # Save if original exists
        cls.original_prompt_exists = cls.actual_judge_prompt_file.exists()
        cls.original_prompt_content = ""
        if cls.original_prompt_exists:
            with open(cls.actual_judge_prompt_file, "r") as f:
                cls.original_prompt_content = f.read()

        with open(cls.actual_judge_prompt_file, "w") as f:
            f.write(TEST_JUDGE_PROMPT_CONTENT)

        # Ensure the main cache dir (if it exists from other runs) is not affected
        # Tests will use TEST_CACHE_DIR by patching CACHE_DIR_BASE

    @classmethod
    def tearDownClass(cls):
        # Restore or remove the dummy judge_prompt.txt
        if cls.original_prompt_exists:
            with open(cls.actual_judge_prompt_file, "w") as f:
                f.write(cls.original_prompt_content)
        else:
            if cls.actual_judge_prompt_file.exists():
                os.remove(cls.actual_judge_prompt_file)

        # Clean up the test cache directory if it was created by any test
        if TEST_CACHE_DIR.parent.exists(): # .test_cache_judge/
             shutil.rmtree(TEST_CACHE_DIR.parent)


    def setUp(self):
        # Clear and ensure test cache directory exists before each test method that uses it
        self._clear_test_cache_directory()
        # MockJudge instances for isolated tests
        self.mock_judge1 = MockJudge(name="TestMock1")
        self.mock_judge2 = MockJudge(name="TestMock2")
        # Reset call counts for mock judges that might be shared or re-used if not careful
        self.mock_judge1._call_count = 0
        self.mock_judge2._call_count = 0


    def _clear_test_cache_directory(self):
        if TEST_CACHE_DIR.exists():
            shutil.rmtree(TEST_CACHE_DIR)
        TEST_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Tests for get_consensus ---
    def test_get_consensus_all_agree_stereotype(self):
        responses = [{"is_stereotype": True, "severity_score": 0.8, "reasoning": "..."},
                       {"is_stereotype": True, "severity_score": 0.6, "reasoning": "..."}]
        label, score = get_consensus(responses)
        self.assertTrue(label)
        self.assertAlmostEqual(score, 0.7)

    def test_get_consensus_majority_stereotype(self):
        responses = [{"is_stereotype": True, "severity_score": 0.8, "reasoning": "..."},
                       {"is_stereotype": False, "severity_score": 0.0, "reasoning": "..."},
                       {"is_stereotype": True, "severity_score": 0.9, "reasoning": "..."}]
        label, score = get_consensus(responses)
        self.assertTrue(label)
        self.assertAlmostEqual(score, 0.85)

    def test_get_consensus_all_agree_no_stereotype(self):
        responses = [{"is_stereotype": False, "severity_score": 0.0, "reasoning": "..."},
                       {"is_stereotype": False, "severity_score": 0.0, "reasoning": "..."}]
        label, score = get_consensus(responses)
        self.assertFalse(label)
        self.assertAlmostEqual(score, 0.0)

    def test_get_consensus_majority_no_stereotype(self):
        responses = [{"is_stereotype": True, "severity_score": 0.8, "reasoning": "..."},
                       {"is_stereotype": False, "severity_score": 0.0, "reasoning": "..."},
                       {"is_stereotype": False, "severity_score": 0.0, "reasoning": "..."}]
        label, score = get_consensus(responses)
        self.assertFalse(label)
        self.assertAlmostEqual(score, 0.0) # Severity is 0.0 because consensus is False

    def test_get_consensus_tie_defaults_to_false(self):
        responses = [{"is_stereotype": True, "severity_score": 0.8, "reasoning": "..."},
                       {"is_stereotype": False, "severity_score": 0.0, "reasoning": "..."}]
        label, score = get_consensus(responses)
        self.assertFalse(label) # Tie (1 True, 1 False) -> False
        self.assertAlmostEqual(score, 0.0) # No True consensus, so severity is 0

    def test_get_consensus_empty_input(self):
        label, score = get_consensus([])
        self.assertFalse(label)
        self.assertAlmostEqual(score, 0.0)

    def test_get_consensus_with_error_responses(self):
        responses = [{"is_stereotype": True, "severity_score": 0.8, "reasoning": "..."},
                       {"error": "API failure"},
                       {"is_stereotype": False, "severity_score": 0.0, "reasoning": "..."}]
        # Valid: 1 True, 1 False. Tie -> False consensus.
        label, score = get_consensus(responses)
        self.assertFalse(label)
        self.assertAlmostEqual(score, 0.0)

    def test_get_consensus_all_errors_or_invalid(self):
        responses = [{"error": "API failure"}, {"foo": "bar"}]
        label, score = get_consensus(responses)
        self.assertFalse(label)
        self.assertAlmostEqual(score, 0.0)

    def test_get_consensus_one_valid_true(self):
        responses = [{"is_stereotype": True, "severity_score": 0.8, "reasoning": "..."},
                       {"error": "API failure"}]
        label, score = get_consensus(responses)
        self.assertTrue(label) # Single valid vote is True
        self.assertAlmostEqual(score, 0.8)

    # --- Tests for Caching Logic ---
    @patch('biaswipe.judge.CACHE_DIR_BASE', TEST_CACHE_DIR) # Patch the global constant for cache dir
    def test_caching_miss_and_hit(self):
        self._clear_test_cache_directory() # Ensure clean cache for this test

        judge = MockJudge(name="CacheTestJudge")
        judge.judge_response = MagicMock(wraps=judge.judge_response) # Wrap to spy on calls but keep functionality

        response_text = "This is a test response for caching."

        # Pass 1: Cache Miss
        result1 = get_judged_response_with_cache(judge, response_text, TEST_JUDGE_PROMPT_CONTENT, TEST_CACHE_DIR)
        judge.judge_response.assert_called_once_with(response_text)
        self.assertIn("is_stereotype", result1)

        # Verify cache file was created
        expected_key = _generate_cache_key(response_text, TEST_JUDGE_PROMPT_CONTENT, judge.name)
        cache_file = TEST_CACHE_DIR / f"{expected_key}.json"
        self.assertTrue(cache_file.exists())

        # Pass 2: Cache Hit
        result2 = get_judged_response_with_cache(judge, response_text, TEST_JUDGE_PROMPT_CONTENT, TEST_CACHE_DIR)
        # Assert judge_response was NOT called again (still called once from the first pass)
        judge.judge_response.assert_called_once()
        self.assertEqual(result1, result2)

    @patch('biaswipe.judge.CACHE_DIR_BASE', TEST_CACHE_DIR)
    def test_caching_key_variation_response_text(self):
        self._clear_test_cache_directory()
        judge = MockJudge(name="CacheKeyVarJudge")
        judge.judge_response = MagicMock(wraps=judge.judge_response)

        response_text1 = "Unique response 1"
        response_text2 = "Unique response 2"

        get_judged_response_with_cache(judge, response_text1, TEST_JUDGE_PROMPT_CONTENT, TEST_CACHE_DIR)
        judge.judge_response.assert_called_once_with(response_text1)

        get_judged_response_with_cache(judge, response_text2, TEST_JUDGE_PROMPT_CONTENT, TEST_CACHE_DIR)
        self.assertEqual(judge.judge_response.call_count, 2)
        judge.judge_response.assert_called_with(response_text2) # Check last call

    @patch('biaswipe.judge.CACHE_DIR_BASE', TEST_CACHE_DIR)
    def test_caching_key_variation_prompt_text(self):
        self._clear_test_cache_directory()
        judge = MockJudge(name="CacheKeyVarPromptJudge")
        judge.judge_response = MagicMock(wraps=judge.judge_response)
        response_text = "Same response, different prompts"

        prompt_text1 = "Prompt version 1 {response_text}"
        prompt_text2 = "Prompt version 2 {response_text}"

        get_judged_response_with_cache(judge, response_text, prompt_text1, TEST_CACHE_DIR)
        judge.judge_response.assert_called_once_with(response_text)

        get_judged_response_with_cache(judge, response_text, prompt_text2, TEST_CACHE_DIR)
        self.assertEqual(judge.judge_response.call_count, 2)

    @patch('biaswipe.judge.CACHE_DIR_BASE', TEST_CACHE_DIR)
    def test_caching_key_variation_judge_name(self):
        self._clear_test_cache_directory()
        judge1 = MockJudge(name="JudgeA")
        judge1.judge_response = MagicMock(wraps=judge1.judge_response)
        judge2 = MockJudge(name="JudgeB") # Different name
        judge2.judge_response = MagicMock(wraps=judge2.judge_response)

        response_text = "Same response, different judges"

        get_judged_response_with_cache(judge1, response_text, TEST_JUDGE_PROMPT_CONTENT, TEST_CACHE_DIR)
        judge1.judge_response.assert_called_once_with(response_text)

        get_judged_response_with_cache(judge2, response_text, TEST_JUDGE_PROMPT_CONTENT, TEST_CACHE_DIR)
        judge2.judge_response.assert_called_once_with(response_text) # Called for its own first time

    @patch('biaswipe.judge.CACHE_DIR_BASE', TEST_CACHE_DIR)
    def test_cache_corrupted_file(self):
        self._clear_test_cache_directory()
        judge = MockJudge(name="CacheCorruptJudge")
        judge.judge_response = MagicMock(wraps=judge.judge_response)
        response_text = "Test for corrupted cache"

        # Manually create a corrupted cache file
        cache_key = _generate_cache_key(response_text, TEST_JUDGE_PROMPT_CONTENT, judge.name)
        cache_file = TEST_CACHE_DIR / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            f.write("this is not valid json")

        self.assertTrue(cache_file.exists())

        # Try to get response, should treat as miss and call judge
        result = get_judged_response_with_cache(judge, response_text, TEST_JUDGE_PROMPT_CONTENT, TEST_CACHE_DIR)
        judge.judge_response.assert_called_once_with(response_text)
        self.assertIn("is_stereotype", result)

        # Check if corrupted file was deleted (or attempted to be) and new one written
        self.assertTrue(cache_file.exists()) # Should exist with new, valid content
        with open(cache_file, "r") as f:
            valid_json_data = json.load(f) # Should load without error
        self.assertEqual(result, valid_json_data)


    # --- Tests for run_judge_ensemble ---
    @patch('biaswipe.judge.CACHE_DIR_BASE', TEST_CACHE_DIR) # run_judge_ensemble uses get_judged_response_with_cache
    def test_run_judge_ensemble_error_handling(self):
        self._clear_test_cache_directory()

        working_judge = MockJudge(name="EnsembleWorking")
        failing_judge = FailingJudge(name="EnsembleFailing")

        judges = [working_judge, failing_judge]
        response_text = "Test ensemble with failure"

        results = run_judge_ensemble(response_text, judges, TEST_JUDGE_PROMPT_CONTENT)

        self.assertEqual(len(results), 2)
        # Check working judge's output (assuming MockJudge's first response)
        self.assertEqual(results[0]['reasoning'], f"{working_judge.name}: Mocked response: Contains stereotypical language regarding profession.")
        self.assertTrue(results[0]['is_stereotype'])

        # Check failing judge's error structure
        self.assertIn("error", results[1])
        self.assertIn(f"Exception for {type(failing_judge).__name__}", results[1]["error"]) # Type name check
        self.assertIn(failing_judge.name, results[1]["details"]) # Specific error message check

    # --- Test API Key Missing for relevant judges ---
    @patch.dict(os.environ, {}, clear=True) # Clear all env vars for this test
    def test_openai_judge_missing_api_key(self):
        with self.assertRaises(MissingApiKeyError) as context:
            OpenAIJudge()
        self.assertIn("OpenAI API key not found", str(context.exception))

    @patch.dict(os.environ, {}, clear=True)
    def test_anthropic_judge_missing_api_key(self):
        with self.assertRaises(MissingApiKeyError) as context:
            AnthropicJudge()
        self.assertIn("Anthropic API key not found", str(context.exception))

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

```
