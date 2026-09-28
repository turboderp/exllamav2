import importlib.util
import unittest
from pathlib import Path


def _load_ngram():
    path = Path(__file__).resolve().parents[1] / "exllamav2" / "generator" / "ngram.py"
    spec = importlib.util.spec_from_file_location("ngram_cache_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NgramCacheTest(unittest.TestCase):
    def test_a_pair_learns_the_last_token(self):
        cache = _load_ngram().NgramCache(1, 4)
        cache.update([7, 8])
        self.assertEqual(cache.predict_next([7], 1, None), 8)

    def test_a_longer_prompt_learns_its_ending(self):
        cache = _load_ngram().NgramCache(1, 4)
        cache.update([10, 20, 30, 99])
        self.assertEqual(cache.predict_next([10, 20, 30], 1, None), 99)
        self.assertEqual(cache.predict_next([1, 2], 1, None), None)

    def test_an_earlier_transition_still_predicts(self):
        cache = _load_ngram().NgramCache(1, 4)
        cache.update([1, 2, 3, 4])
        self.assertEqual(cache.predict_next([1, 2], 1, None), 3)


if __name__ == "__main__":
    unittest.main()
