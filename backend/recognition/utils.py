import time
from collections import deque, Counter

class SignProcessor:
    def __init__(self):
        # store recent predictions
        self.buffer = deque(maxlen=20)

        # output state
        self.text = ""
        self.last_letter = None

        # timing
        self.last_emit_time = time.time()
        self.last_seen_time = time.time()

        # parameters
        self.STABLE_FRAMES = 5      # how many consistent frames needed
        self.COOLDOWN = 0.8         # seconds between letters
        self.SPACE_TIMEOUT = 1.5    # pause → space

    def update(self, prediction):
        now = time.time()

        # track activity
        if prediction:
            self.last_seen_time = now

        # add prediction to buffer
        if prediction is not None:
            self.buffer.append(prediction)

        # stable prediction
        letter = self._get_stable_prediction()

        # case 1: no stable prediction, a space
        if letter is None:
            if now - self.last_seen_time > self.SPACE_TIMEOUT:
                self._add_space()
            return self.text

        # case 2: avoid fast duplicates
        if now - self.last_emit_time < self.COOLDOWN:
            return self.text

        # accept new letter
        self.last_letter = letter
        self.last_emit_time = now

        # handle gestures
        if letter == "DELETE":
            self._delete_last()
            return self.text

        self.text += letter

        return self.text

    def _get_stable_prediction(self):
        if len(self.buffer) < self.STABLE_FRAMES:
            return None

        counts = Counter(self.buffer)
        most_common, freq = counts.most_common(1)[0]

        if freq >= self.STABLE_FRAMES:
            return most_common

        return None

    def _add_space(self):
        if not self.text.endswith(" "):
            self.text += " "

    def _delete_last(self):
        self.text = self.text[:-1]