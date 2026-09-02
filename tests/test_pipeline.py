"""Invariants that must never break: no peeking, no repeats, valid submissions."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest

from hangman.encoding import ALPHABET, MASK_ID, board_tokens, encode_words, render
from hangman.policies import LengthFrequencyPolicy
from hangman.policies.base import Policy
from hangman.simulator import play
from hangman.submit import validate_submission, write_submission

WORDS = ["cat", "hangman", "bookkeeper", "a-frame", "zzz"]


def test_board_never_leaks_hidden_letters():
    chars, _, present = encode_words(WORDS, max_len=12)
    revealed = np.zeros((len(WORDS), 26), dtype=bool)
    revealed[:, ord("a") - 97] = True
    tokens = board_tokens(chars, revealed)
    for i, word in enumerate(WORDS):
        for j, ch in enumerate(word):
            if ch == "a":
                assert tokens[i, j] == ord("a") - 97 + 2
            elif ch.isalpha():
                assert tokens[i, j] == MASK_ID, f"{word!r} leaked {ch!r}"


def test_render_matches_board():
    chars, _, _ = encode_words(["hangman"], max_len=8)
    revealed = np.zeros((1, 26), dtype=bool)
    revealed[0, ord("n") - 97] = True
    assert render(chars[0], revealed[0]) == "__n___n"


def test_non_letters_are_visible_from_the_start():
    chars, _, _ = encode_words(["a-frame"], max_len=8)
    revealed = np.zeros((1, 26), dtype=bool)
    assert render(chars[0], revealed[0]) == "_?_____"


def test_no_repeated_letters_in_output():
    policy = LengthFrequencyPolicy(["hello", "world", "hangman", "python"])
    result = play(WORDS, policy, max_len=12)
    for seq in result.guesses:
        assert len(set(seq)) == len(seq)
        assert seq.isalpha()


def test_policy_cannot_see_the_word():
    """A policy that only ever guesses 'z' must lose every z-less word."""

    class OnlyZ(Policy):
        def scores(self, obs):
            s = np.zeros((len(obs), 26))
            s[:, ord("z") - 97] = 1.0
            return s

    result = play(["cat", "dog", "zzz"], OnlyZ(), max_len=6)
    assert list(result.won) == [False, False, True]
    assert result.wrong[0] == 6


def test_submission_validation(tmp_path):
    path = write_submission(["abc", "de"], tmp_path / "s.csv")
    validate_submission(path, expected_rows=2)
    write_submission(["aab"], tmp_path / "bad.csv")
    with pytest.raises(AssertionError):
        validate_submission(tmp_path / "bad.csv", expected_rows=1)
