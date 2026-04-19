from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class TimerEvent:
    """Result from one timing update."""

    commit_letter: Optional[str] = None
    commit_space: bool = False


class GestureTimer:
    """
    Time-based gesture gate for text entry.

    Rules implemented:
    - Single letter: a label must stay stable for letter_hold_ms before commit.
    - Double letter: repeating the same label requires a pause of repeat_pause_ms
      after the previous release.
    - Space: if no hand is present for word_pause_ms, emit one space event.

    This class supports two update styles:
    - update(..., timestamp_ms=<explicit time>)
    - update_frame(...) using fixed frame_ms increments (default 300ms)
    """

    def __init__(
        self,
        frame_ms: int = 300,
        letter_hold_ms: int = 600,
        repeat_pause_ms: int = 450,
        word_pause_ms: int = 900,
    ) -> None:
        self.frame_ms = frame_ms
        self.letter_hold_ms = letter_hold_ms
        self.repeat_pause_ms = repeat_pause_ms
        self.word_pause_ms = word_pause_ms

        self._clock_ms = 0

        self._active_label: Optional[str] = None
        self._active_label_since_ms: Optional[int] = None
        self._committed_in_active_streak = False
        self._repeat_allowed_for_active_streak = True

        self._last_committed_label: Optional[str] = None
        self._last_release_ms_by_label: Dict[str, int] = {}

        self._no_hand_since_ms: Optional[int] = None
        self._space_emitted_for_current_gap = False

    def _release_active_label(self, timestamp_ms: int) -> None:
        if self._active_label is not None:
            self._last_release_ms_by_label[self._active_label] = timestamp_ms

        self._active_label = None
        self._active_label_since_ms = None
        self._committed_in_active_streak = False
        self._repeat_allowed_for_active_streak = True

    def _start_new_label_streak(self, label: str, timestamp_ms: int) -> None:
        self._active_label = label
        self._active_label_since_ms = timestamp_ms
        self._committed_in_active_streak = False

        # If this is a repeat of the most recently committed label,
        # require a release pause before allowing a second commit.
        if label == self._last_committed_label:
            release_ms = self._last_release_ms_by_label.get(label)
            self._repeat_allowed_for_active_streak = (
                release_ms is not None
                and (timestamp_ms - release_ms) >= self.repeat_pause_ms
            )
        else:
            self._repeat_allowed_for_active_streak = True

    def update(
        self,
        label: Optional[str],
        hand_present: bool,
        timestamp_ms: int,
    ) -> TimerEvent:
        event = TimerEvent()

        # Hand-down timing controls spacing / word completion.
        if not hand_present:
            self._release_active_label(timestamp_ms)

            if self._no_hand_since_ms is None:
                self._no_hand_since_ms = timestamp_ms

            no_hand_elapsed = timestamp_ms - self._no_hand_since_ms
            if (
                no_hand_elapsed >= self.word_pause_ms
                and not self._space_emitted_for_current_gap
            ):
                event.commit_space = True
                self._space_emitted_for_current_gap = True

            return event

        # Hand is present: reset no-hand gap state.
        self._no_hand_since_ms = None
        self._space_emitted_for_current_gap = False

        # No usable label yet (for example low confidence frame).
        if not label:
            self._release_active_label(timestamp_ms)
            return event

        # Start or switch active streak.
        if self._active_label != label:
            self._release_active_label(timestamp_ms)
            self._start_new_label_streak(label, timestamp_ms)

        # Stable-hold timing.
        held_ms = timestamp_ms - int(self._active_label_since_ms)
        meets_hold = held_ms >= self.letter_hold_ms

        if (
            meets_hold
            and not self._committed_in_active_streak
            and self._repeat_allowed_for_active_streak
        ):
            event.commit_letter = label
            self._committed_in_active_streak = True
            self._last_committed_label = label

        return event

    def update_frame(self, label: Optional[str], hand_present: bool) -> TimerEvent:
        """Advance by one fixed frame and process timing."""
        event = self.update(label=label, hand_present=hand_present, timestamp_ms=self._clock_ms)
        self._clock_ms += self.frame_ms
        return event
