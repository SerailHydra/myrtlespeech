import torch
import pytest
import hypothesis.strategies as st
from hypothesis import assume, given

from myrtlespeech.post_process.ctc_beam_decoder import CTCBeamDecoder


# Fixtures and Strategies -----------------------------------------------------


# Tests -----------------------------------------------------------------------

# Unit Tests ----------------------------------------


def test_simple_2x2() -> None:
    """Ensures simple beam search over 2x2 output beats naive greedy method.

    Let the output for a single example in a batch be:

        | symbol | p(symbol @ t = 0) | p(symbol @ t = 1) |
        |--------|-------------------|-------------------|
        | "a"    | 0.3               | 0.3               |
        | blank  | 0.7               | 0.7               |

    A naive approach will predict blank as the most likely sequence. The beam
    search should find p(a).

    Why?

    p(blank) = p(blank, blank)
             = 0.7 * 0.7
             = 0.49

    p(a) = p(a, blank) + p(blank, a) + p(a, a)
         = 0.21 + 0.21 + 0.09
         = 0.51
    """
    alphabet = {"a": 0, "_": 1}  # '_' represents "blank"
    # create tensor with grid described in docstring
    x = torch.empty((2, 1, len(alphabet)))  # seq_len=2, batch=1
    x[:, 0, alphabet["a"]] = torch.tensor([0.3, 0.3])
    x[:, 0, alphabet["_"]] = torch.tensor([0.7, 0.7])

    lengths = torch.tensor([2], dtype=torch.int8)

    ctc_decoder = CTCBeamDecoder(
        blank_index=alphabet["_"],
        beam_width=2,
        prune_threshold=0.0,
        separator_index=None,
        language_model=None,
    )

    result = ctc_decoder(x, lengths)

    assert result[0] == [alphabet["a"]]


def test_simple_lanaguage_model() -> None:
    """Ensures beam search with language model corrects homophones.

    Without a language model the test should find "do" as the most likely. We
    create two language models that return 2.0 (invalid probability, yes) for
    both "dew" and "due" respectively and 0.0 for all other words.

    The test ensures that "dew" and "due" are returned when their respective
    language models are used.
    """
    alphabet = dict(zip("deouw_ ", range(7)))
    x = torch.empty((4, 1, len(alphabet)))
    x[:, 0, alphabet["d"]] = torch.tensor([0.75, 0.05, 0.10, 0.01])
    x[:, 0, alphabet["e"]] = torch.tensor([0.05, 0.20, 0.20, 0.01])
    x[:, 0, alphabet["o"]] = torch.tensor([0.05, 0.30, 0.35, 0.01])
    x[:, 0, alphabet["u"]] = torch.tensor([0.05, 0.20, 0.10, 0.01])
    x[:, 0, alphabet["w"]] = torch.tensor([0.05, 0.00, 0.20, 0.01])
    x[:, 0, alphabet["_"]] = torch.tensor([0.00, 0.00, 0.10, 0.94])
    x[:, 0, alphabet[" "]] = torch.tensor([0.05, 0.05, 0.05, 0.01])

    lengths = torch.tensor([x.size()[0]], dtype=torch.int8)

    def lm(target: str):
        target_tuple = tuple(alphabet[c] for c in target) + (alphabet[" "],)
        return lambda w: 2.0 if w == target_tuple else 0.0

    # first check no LM returns "do"
    ctc_decoder = CTCBeamDecoder(blank_index=alphabet["_"], beam_width=20)
    assert ctc_decoder(x, lengths) == [[alphabet[c] for c in "do"]]

    # check LM corrects this to dew or due
    for target in ["dew", "due"]:
        ctc_decoder = CTCBeamDecoder(
            blank_index=alphabet["_"],
            beam_width=20,
            separator_index=alphabet[" "],
            language_model=lm(target),
            lm_weight=10.0,
            word_weight=2.0,
        )

        assert ctc_decoder(x, lengths) == [[alphabet[c] for c in target + " "]]


@given(dtype=st.sampled_from([torch.half, torch.float, torch.double]))
def test_ctc_beam_decoder_raises_value_error_for_float_dtypes(
    dtype: torch.dtype
) -> None:
    """Ensures ValueError raised when lengths.dtype is float."""
    x = torch.empty((5, 1, 3))
    lengths = torch.tensor([5], dtype=dtype)
    ctc_decoder = CTCBeamDecoder(blank_index=0, beam_width=20)
    with pytest.raises(ValueError):
        ctc_decoder(x, lengths)


@given(x_batch_size=st.integers(1, 32), lengths_batch_size=st.integers(1, 32))
def test_ctc_beam_decoder_raises_value_error_when_batch_x_lengths_differ(
    x_batch_size: int, lengths_batch_size: int
) -> None:
    """Ensures ValueError raised when batch size of x and lengths differs."""
    assume(x_batch_size != lengths_batch_size)

    ctc_decoder = CTCBeamDecoder(blank_index=0, beam_width=20)

    # create input tensors, batch and alphabet size fixed to 10 and 5
    x = torch.empty((10, x_batch_size, 5))
    lengths = torch.empty(lengths_batch_size, dtype=torch.int16)

    with pytest.raises(ValueError):
        ctc_decoder(x, lengths)


def test_ctc_beam_decoder_raises_value_error_lengths_values_greater_seq_len() -> None:
    """Ensures ValueError when lengths entry is greater than seq len of x."""
    ctc_decoder = CTCBeamDecoder(blank_index=0, beam_width=20)

    x = torch.empty((10, 3, 20))
    lengths = torch.tensor([1, 19, 1], dtype=torch.uint8)

    with pytest.raises(ValueError):
        ctc_decoder(x, lengths)


def test_ctc_beam_decoder_raises_value_error_lm_but_no_lm_weight() -> None:
    """Ensures ValueError raised when language_model set but not lm_weight."""
    with pytest.raises(ValueError):
        CTCBeamDecoder(
            blank_index=0, beam_width=20, language_model=lambda _: 0.0
        )
