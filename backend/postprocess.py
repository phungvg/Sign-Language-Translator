"""
Post processing
Handle:
(1) Debounce Buffer: suprress duplicate letters (Ex: Hello) (Later add ryan time counter)
(2) Suggest completion: fix misselled words (Ex: Helo)

Ex
(1) User signs: H E L L O
Without debounce: H E L L L L L L O  : holding L too long
With debounce:    H E L [reset] L O  : clean double letter

(2)
Input:  "hel" : current letters signed so far
Output: ["hello", "help", "held", "helm", "hell"]   top 5 words starting with "hel"
"""
from spellchecker import SpellChecker

#Debounce Buffer
class DebounceBuffer:
    def __init__(self):    
        self.last_char = None   # last character that was emitted

    def push(self, char):
        """
        Check if char is duplicate.

        Args:
            char: predicted letter or digit

        Returns:
            char  -> if new character (emit)
            None  -> if duplicate (suppress)
        """
        if char == self.last_char:
            return None         # same as last -> suppress

        self.last_char = char   # new letter -> update and emit
        return char

    def reset(self):
        """
        Reset buffer — allows same letter to be emitted again.
        Called when hand briefly rests between double letters.
        Example: signing LL in HELLO -> sign L, rest, sign L
        """
        self.last_char = None

#-----------------------------------------------------------------------------------
spell = SpellChecker()
def suggest_completions(prefix, n=5):
    if not prefix:
        return []

    prefix = prefix.lower().strip()

    # Search spell checker word list for words starting with prefix
    matches = [
        word for word in spell.word_frequency.dictionary
        if word.startswith(prefix)
    ]

    if not matches:
        return []

    # Sort by word frequency — most common words first
    matches.sort(key=lambda w: spell.word_usage_frequency(w), reverse=True)

    return matches[:n]
#-----------------------------------------------------------------------------------

# Quick test at bottom of postprocess.py
# if __name__ == "__main__":
#     # Test suggest_completions
#     results = suggest_completions("ea")
#     print(f"ea -> {results}")
#     assert len(results) <= 5
#     assert all(w.startswith("ea") for w in results)

#     results = suggest_completions("ban")
#     print(f"ban  -> {results}")
#     assert len(results) <= 5
#     assert all(w.startswith("ban") for w in results)

#     results = suggest_completions("")
#     assert results == []

