"""Submission writing and validation."""
from __future__ import annotations

import csv
from pathlib import Path


def write_submission(guesses: list[str], path: str | Path = "submission.csv") -> Path:
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["word_id", "guessed_letters_string"])
        for word_id, seq in enumerate(guesses):
            writer.writerow([word_id, seq])
    return path


def validate_submission(path: str | Path, expected_rows: int = 250_000) -> None:
    """Fail loudly on the mistakes that silently cost points."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    header, body = rows[0], rows[1:]
    assert header == ["word_id", "guessed_letters_string"], f"bad header: {header}"
    assert len(body) == expected_rows, f"{len(body)} rows, expected {expected_rows}"

    for i, (word_id, seq) in enumerate(body):
        assert int(word_id) == i, f"word_id {word_id} out of order at line {i}"
        assert seq.isalpha() and seq.islower(), f"row {i}: bad characters {seq!r}"
        assert len(set(seq)) == len(seq), f"row {i}: repeated letter in {seq!r}"
    print(f"submission OK: {len(body):,} rows, no repeats, ids aligned")
