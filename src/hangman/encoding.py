"""Character-level encoding for Hangman board states.

Token vocabulary (size 29):
    0  PAD      padding beyond the end of the word
    1  MASK     a hidden letter (rendered as '_' on the board)
    2..27       a revealed letter, id = 2 + ord(c) - ord('a')
    28 OTHER    a revealed non-letter character (space, hyphen, digit, ...)

Non-letter characters are visible from the start of the game and are never
guessable, so they collapse into a single OTHER token.
"""
from __future__ import annotations

import numpy as np

ALPHABET = "abcdefghijklmnopqrstuvwxyz"
N_LETTERS = len(ALPHABET)

PAD_ID = 0
MASK_ID = 1
LETTER_OFFSET = 2
OTHER_ID = LETTER_OFFSET + N_LETTERS  # 28
VOCAB_SIZE = OTHER_ID + 1             # 29

MAX_LIVES = 6

# Sentinels used in the packed character array (see encode_words).
PAD_CHAR = -1     # position does not exist
OTHER_CHAR = -2   # position holds a visible non-letter

_LETTER_TO_ID = {c: i for i, c in enumerate(ALPHABET)}


def encode_words(words: list[str], max_len: int | None = None):
    """Pack words into a dense integer matrix.

    Returns
    -------
    chars : (n, max_len) int8
        Letter index 0..25, or PAD_CHAR / OTHER_CHAR sentinels.
    lengths : (n,) int32
    present : (n, 26) bool
        Whether each letter occurs anywhere in the word.
    """
    if max_len is None:
        max_len = max(len(w) for w in words)

    n = len(words)
    chars = np.full((n, max_len), PAD_CHAR, dtype=np.int8)
    lengths = np.zeros(n, dtype=np.int32)
    present = np.zeros((n, N_LETTERS), dtype=bool)

    for i, word in enumerate(words):
        if len(word) > max_len:
            raise ValueError(f"word {word!r} exceeds max_len={max_len}")
        lengths[i] = len(word)
        for j, ch in enumerate(word):
            idx = _LETTER_TO_ID.get(ch)
            if idx is None:
                chars[i, j] = OTHER_CHAR
            else:
                chars[i, j] = idx
                present[i, idx] = True
    return chars, lengths, present


def board_tokens(chars: np.ndarray, revealed: np.ndarray) -> np.ndarray:
    """Render packed words as model input tokens given revealed letters.

    Parameters
    ----------
    chars : (b, max_len) int8   packed characters
    revealed : (b, 26) bool     letters that have been correctly guessed

    Returns
    -------
    (b, max_len) int64 token ids. Hidden letters become MASK; the true letter
    is never exposed for a position whose letter has not been revealed.
    """
    safe = np.where(chars >= 0, chars, 0).astype(np.intp)
    is_revealed = np.take_along_axis(revealed, safe, axis=1)

    tokens = np.full(chars.shape, MASK_ID, dtype=np.int64)
    tokens = np.where(is_revealed & (chars >= 0), safe + LETTER_OFFSET, tokens)
    tokens = np.where(chars == OTHER_CHAR, OTHER_ID, tokens)
    tokens = np.where(chars == PAD_CHAR, PAD_ID, tokens)
    return tokens


def render(chars_row: np.ndarray, revealed_row: np.ndarray) -> str:
    """Human-readable board for a single game, for logging and tests."""
    out = []
    for c in chars_row:
        if c == PAD_CHAR:
            break
        if c == OTHER_CHAR:
            out.append("?")
        elif revealed_row[c]:
            out.append(ALPHABET[c])
        else:
            out.append("_")
    return "".join(out)
