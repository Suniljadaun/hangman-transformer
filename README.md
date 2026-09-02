# Hangman Agent — masked-word transformer with on-policy training

A Hangman-playing agent for the Meltwater *Brand & Buzzword* hackathon. Six wrong
guesses per word, a 250,000-word secret test set, and a training vocabulary that
does **not** contain the test words — so the agent has to learn English structure,
not a word list.

## Approach in one paragraph

The board is a character sequence with most letters hidden. A transformer encoder
reads it, conditioned on the letters already tried — including, crucially, the ones
known to be **absent**, which carry as much information as the reveals. Two heads
predict the next guess: a masked-language-model head that predicts the letter at
each hidden position (aggregated across the word by noisy-OR), and a pooled
bag-of-letters head that answers "does this letter appear anywhere" directly. The
network is trained not on randomly masked words but on the board states the policy
actually reaches, collected by replaying games with the current model and
aggregating across rounds (DAgger).

## Layout

```
src/hangman/
  encoding.py     token vocabulary, packing, board rendering
  data.py         word-list loading, auditing, disjoint holdout split
  simulator.py    batched no-peeking game simulator  <- the secret word lives here and nowhere else
  model.py        HangmanNet: transformer + MLM head + word head
  states.py       on-policy state collection (9 bytes per state) and the torch Dataset
  train.py        AdamW + cosine schedule + AMP training loop
  evaluate.py     competition metrics and a per-length breakdown
  submit.py       submission writing and the validator that catches silent point leaks
  policies/
    base.py       Policy interface; guarantees a letter is never repeated
    frequency.py  length-conditioned letter frequency (the floor)
    ngram.py      character n-gram back-off with noisy-OR aggregation
    dictionary.py pattern-filtered candidate voting, memoised on board state
    neural.py     the learned policy
    ensemble.py   score fusion and the exploration wrapper used for state collection
notebooks/
  01_train.ipynb  Kaggle GPU: DAgger rounds, checkpoints
  02_submit.ipynb Kaggle: inference only, writes submission.csv
tests/            invariants: no peeking, no repeated letters, valid submission
```

## Measured baselines

On a 2,000-word holdout disjoint from training (a `web2` proxy for `train.txt`,
same shape: 234k words, lengths 3–20):

| policy | win rate | mean strikes |
|---|---|---|
| length-conditioned frequency | 15.6% | 5.67 |
| pattern-filtered dictionary | 17.5% | 5.60 |
| character n-gram back-off | **54.3%** | 4.44 |

The dictionary policy barely beats frequency, which is the whole lesson of this
competition: the test words are not in the training vocabulary, so anything that
relies on recognising the word collapses. The n-gram model generalises because it
only needs to have seen the *neighbourhoods*. The transformer is the same idea with
a learned, longer-range notion of neighbourhood.

## Running it

```bash
pip install -r requirements.txt
python -m pytest tests -q

# baselines and a holdout report
python -c "
import sys; sys.path.insert(0,'src')
from hangman.data import load_words, split_holdout
from hangman.policies.ngram import NGramPolicy
from hangman.evaluate import evaluate, print_report
train, holdout = split_holdout(load_words('data/train.txt'), n_holdout=10000)
print_report(evaluate(holdout, NGramPolicy(train)))
"
```

Training runs on Kaggle (see `notebooks/01_train.ipynb`). Everything is reproducible
from `train.txt` alone; `test.txt` is used only to simulate games, never to fit
anything.

## Integrity

- The simulator is the only object that holds a secret word. Policies receive an
  `Observation` containing the masked board, the guess history and lives remaining.
  There is no code path that puts a word into an `Observation`.
- `test.txt` is never used to build a vocabulary, a frequency table, or an n-gram
  count. Training and holdout words come from `train.txt` only.
- No per-word special-casing and no cached answers: the policy is a pure function
  of the observation.
- `submit.validate_submission` fails loudly on repeated letters, misaligned
  `word_id`s, and wrong row counts.
