import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ambiguity_disagreement_episodes import direct_similarity
from ambiguity_induced_relocation import burst_events, frame_index
from matchmaker_ambiguity_rescue import rolling_state, zoh
from online_relocation_validation import latest


class TestMatchmakerAmbiguityRescue(unittest.TestCase):
    def test_zoh_never_uses_future_output(self):
        path = np.array([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]])
        query = np.array([0.5, 1.0, 1.9, 2.0, 3.9, 4.0])
        position, update = zoh(path, query)
        self.assertTrue(np.isnan(position[0]))
        np.testing.assert_allclose(position[1:], [10, 10, 20, 20, 40])
        np.testing.assert_allclose(update[1:], [1, 1, 2, 2, 4])

    def test_rolling_state_is_prefix_invariant(self):
        position = np.array([1.0, 1.2, 1.4, 1.3, 2.0])
        update = np.array([0.8, 1.8, 2.8, 3.8, 4.8])
        time_axis = np.arange(1.0, 6.0)
        full = rolling_state(position, update, time_axis)
        prefix = rolling_state(position[:4], update[:4], time_axis[:4])
        np.testing.assert_allclose(full[:4], prefix, atol=0, rtol=0)

    def test_direct_similarity_uses_pinned_forward_windows(self):
        chroma = np.zeros((45, 12))
        chroma[:20, 0] = 1
        chroma[20:40, 0] = 1
        result = direct_similarity(chroma, np.array([0.1, 0.1]), np.array([2.1, 3.1]))
        self.assertAlmostEqual(result[0], 1.0)
        self.assertTrue(np.isnan(result[1]))

    def test_relocation_bursts_keep_largest_jump(self):
        path = np.array(
            [[9.0, 10.0, 10.5, 13.0, 14.0], [0.0, 3.0, 9.0, 12.0, 12.2]]
        )
        self.assertEqual(burst_events(path), [2, 3])

    def test_score_window_index_rejects_boundaries(self):
        self.assertIsNone(frame_index(0.0, 10))
        self.assertEqual(frame_index(0.1, 10), 0)
        self.assertIsNone(frame_index(2.0, 10))

    def test_latest_relocation_position_is_causal(self):
        path = np.array([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]])
        self.assertIsNone(latest(path, 0.9))
        self.assertEqual(latest(path, 3.9), 20.0)
        self.assertEqual(latest(path, 4.0), 40.0)


if __name__ == "__main__":
    unittest.main()
