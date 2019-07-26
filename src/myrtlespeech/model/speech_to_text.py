from typing import Callable, List, Optional, Tuple, Union

import torch

from myrtlespeech.data.alphabet import Alphabet
from myrtlespeech.model.encoder_decoder import EncoderDecoder
from myrtlespeech.post_process.ctc_beam_decoder import CTCBeamDecoder
from myrtlespeech.post_process.ctc_greedy_decoder import CTCGreedyDecoder


class SpeechToText(torch.nn.Module):
    """A speech to text model.

    .. todo::

        Document this

    Args:
        encoder:
        decoder:
        loss: Callable that takes [log_probs, targets, input_lengths,
            target_lengths]?
    """

    def __init__(
        self,
        alphabet: Alphabet,
        model: EncoderDecoder,
        loss: torch.nn.CTCLoss,
        pre_process_steps: List[Tuple[Callable, bool]],
        post_process: Union[None, CTCGreedyDecoder, CTCBeamDecoder],
    ):
        super().__init__()
        self.alphabet = alphabet
        self.model = model
        self.loss = loss
        self.pre_process_steps = pre_process_steps
        self.post_process = post_process

    def forward(
        self, x: torch.Tensor, seq_lens: Optional[torch.Tensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """TODO

        Arguments:
            x: Tuple of [input, input lengths]

        Returns:
            Tuple of [unnorm log probs, unnorm log prob lengths]
        """
        return self.model(x, seq_lens=seq_lens)

    @property
    def transform(self) -> Callable:
        """
        TODO

        Returns a callable that takes a tensor of size [audio_len], dtype int
        of audio data and returns tensor of size [channels, features, seq_len],
        dtype float32
        """

        def transform(x):
            for step, train_only in self.pre_process_steps:
                if train_only and not self.training:
                    continue
                x = step(x)
            return x

        return transform
