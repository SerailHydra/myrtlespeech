"""Builds an :py:class:`.EncoderDecoder` model from a configuration."""
from typing import Optional

from myrtlespeech.builders.decoder import build as build_decoder
from myrtlespeech.builders.encoder import build as build_encoder
from myrtlespeech.model.encoder_decoder import EncoderDecoder
from myrtlespeech.protos import encoder_decoder_pb2


def build(
    encoder_decoder_cfg: encoder_decoder_pb2.EncoderDecoder,
    input_features: int,
    output_features: int,
    input_channels: Optional[int],
    seq_len_support: bool = False,
) -> EncoderDecoder:
    """Returns a :py:class:`.EncoderDecoder` model based on the model config.

    Args:
        encoder_decoder_cfg: A ``EncoderDecoder.proto`` object containing the
            config for the desired :py:class:`.EncoderDecoder`.

        input_features: The number of features for the input.

        input_channels: The number of channels for the input. May be ``None``
            if encoder does require it.

        seq_len_support: If :py:data:`True`, the returned encoder and decoder
            :py:meth:`torch.nn.Module.forward` methods must optionally accept a
            ``seq_lens`` kwarg.

    Returns:
        An :py:class:`.EncoderDecoder` based on the config.

    Raises:
        :py:class:`ValueError`: On invalid configuration.
    """
    encoder, input_features = build_encoder(
        encoder_decoder_cfg.encoder,
        input_features=input_features,
        input_channels=input_channels,
        seq_len_support=seq_len_support,
    )

    decoder = build_decoder(
        encoder_decoder_cfg.decoder,
        input_features=input_features,
        output_features=output_features,
        seq_len_support=seq_len_support,
    )

    return EncoderDecoder(encoder=encoder, decoder=decoder)
