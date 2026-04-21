import importlib.util
from pathlib import Path
import unittest


_TIMER_PATH = Path(__file__).with_name("timer.py")
_TIMER_SPEC = importlib.util.spec_from_file_location("timer", _TIMER_PATH)
if _TIMER_SPEC is None or _TIMER_SPEC.loader is None:
    raise RuntimeError(f"Unable to load timer module from {_TIMER_PATH}")

_TIMER_MODULE = importlib.util.module_from_spec(_TIMER_SPEC)
_TIMER_SPEC.loader.exec_module(_TIMER_MODULE)
GestureTimer = _TIMER_MODULE.GestureTimer


class GestureTimerTests(unittest.TestCase):
    def _run_frames(self, timer, frames):
        """
        frames: list of tuples (label, hand_present, count)
        Returns all emitted (commit_letter, commit_space) events.
        """
        events = []
        for label, hand_present, count in frames:
            for _ in range(count):
                evt = timer.update_frame(label=label, hand_present=hand_present)
                if evt.commit_letter or evt.commit_space:
                    events.append((evt.commit_letter, evt.commit_space))
        return events

    def test_single_letter_requires_hold_and_commits_once(self):
        timer = GestureTimer(
            frame_ms=300,
            letter_hold_ms=900,
            repeat_pause_ms=600,
            word_pause_ms=1500,
        )

        # 4 frames at 300ms -> first eligible commit at 900ms.
        events = self._run_frames(timer, [("A", True, 4)])
        self.assertEqual(events, [("A", False)])

        # Keep holding same letter; should not auto-repeat.
        events = self._run_frames(timer, [("A", True, 4)])
        self.assertEqual(events, [])

    def test_double_letter_requires_pause_then_rehold(self):
        timer = GestureTimer(
            frame_ms=300,
            letter_hold_ms=900,
            repeat_pause_ms=600,
            word_pause_ms=1500,
        )

        # First A commit.
        events = self._run_frames(timer, [("A", True, 4)])
        self.assertEqual(events, [("A", False)])

        # Pause is too short (1 frame = 300ms), then re-hold A.
        # Repeat should stay blocked for this streak.
        events = self._run_frames(timer, [(None, False, 1), ("A", True, 4)])
        self.assertEqual(events, [])

        # Long enough pause (2 frames = 600ms), then re-hold A -> second commit allowed.
        events = self._run_frames(timer, [(None, False, 2), ("A", True, 4)])
        self.assertEqual(events, [("A", False)])

    def test_space_emits_once_for_long_hand_down_gap(self):
        timer = GestureTimer(
            frame_ms=300,
            letter_hold_ms=900,
            repeat_pause_ms=600,
            word_pause_ms=1500,
        )

        # 6 frames with no hand crosses 1500ms threshold once.
        events = self._run_frames(timer, [(None, False, 6)])
        self.assertEqual(events, [(None, True)])

        # Keep hand down; should not emit repeated spaces in same gap.
        events = self._run_frames(timer, [(None, False, 6)])
        self.assertEqual(events, [])

        # Hand returns, then another long gap should emit one new space.
        events = self._run_frames(timer, [("B", True, 2), (None, False, 6)])
        self.assertEqual(events, [(None, True)])


if __name__ == "__main__":
    unittest.main()
