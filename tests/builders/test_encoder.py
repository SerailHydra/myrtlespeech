from typing import Optional

import pytest
import torch
import hypothesis.strategies as st
from hypothesis import assume, given

from myrtlespeech.builders.encoder import build
from myrtlespeech.model.encoder.encoder import Encoder
from myrtlespeech.model.encoder.vgg import vgg_output_size
from myrtlespeech.protos import encoder_pb2
from tests.builders.test_rnn import rnn_module_match_cfg
from tests.builders.test_vgg import vgg_module_match_cfg
from tests.protos.test_encoder import encoders


# Utilities -------------------------------------------------------------------


def encoder_module_match_cfg(
    encoder: Encoder,
    encoder_cfg: encoder_pb2.Encoder,
    input_features: int,
    input_channels: Optional[int],
) -> None:
    """Ensures Encoder matches protobuf configuration."""
    assert isinstance(encoder, Encoder)

    # check cnn config
    if encoder_cfg.HasField("no_cnn"):
        assert encoder.cnn is None
    elif encoder_cfg.HasField("vgg"):
        vgg_module_match_cfg(  # type: ignore
            encoder.cnn, encoder_cfg.vgg, input_channels
        )
        out_size = vgg_output_size(
            encoder.cnn, torch.Size([-1, input_channels, input_features, -1])
        )
        input_features = out_size[1] * out_size[2]
    else:
        raise ValueError("expected either no_cnn or vgg to be set")

    # check rnn config
    if encoder_cfg.HasField("no_rnn"):
        assert encoder.rnn is None
    elif encoder_cfg.HasField("rnn"):
        rnn_module_match_cfg(encoder.rnn, encoder_cfg.rnn, input_features)
    else:
        raise ValueError("expected either no_rnn or rnn to be set")


# Tests -----------------------------------------------------------------------


@given(
    encoder_cfg=encoders(),
    input_features=st.integers(min_value=1, max_value=32),
    input_channels=st.one_of(st.none(), st.integers(min_value=1, max_value=8)),
)
def test_build_encoder_returns_correct_module_structure_and_out_features(
    encoder_cfg: encoder_pb2.Encoder,
    input_features: int,
    input_channels: Optional[int],
) -> None:
    """Ensures tuple returned by ``build`` has correct structure."""
    assume(encoder_cfg.HasField("no_cnn") or input_channels is not None)
    encoder, _ = build(encoder_cfg, input_features, input_channels)
    encoder_module_match_cfg(
        encoder, encoder_cfg, input_features, input_channels
    )


@given(
    encoder_cfg=encoders(),
    input_features=st.integers(min_value=1, max_value=32),
)
def test_input_channels_none_when_not_no_cnn_raises_value_error(
    encoder_cfg: encoder_pb2.Encoder, input_features: int
) -> None:
    """Ensures ValueError raised when input_channels is None, cnn not None."""
    assume(not encoder_cfg.HasField("no_cnn"))
    with pytest.raises(ValueError):
        build(encoder_cfg, input_features, None)
