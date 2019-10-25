from typing import Callable
from typing import Optional
from typing import Tuple

import torch


class RNNT(torch.nn.Module):
    r"""`RNN-T <https://arxiv.org/pdf/1211.3711.pdf>`_ Network. Architecture
    based on `Streaming End-to-end Speech Recognition For Mobile Devices
    <https://arxiv.org/pdf/1811.06621.pdf>`_.

    Args:

        encoder: A :py:class:`RNNTEncoder` with initialised RNN-T encoder.

            Must accept as input a tuple where the first element is the network
            input (a :py:`torch.Tensor`) with size ``[max_seq_len, batch,
            in_features]`` and the second element is a
            :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represents the sequence length of the corresponding *input*
            sequence to the rnn.

            It must return a tuple where the first element is the result after
            applying the module to the input. It must have size
            ``[max_out_seq_len, batch, rnn_features]``. The second element of
            the tuple return value is a :py:class:`torch.Tensor` with size
            ``[batch]`` where each entry represents the sequence length of the
            corresponding *output* sequence. These may be different than the
            input sequence lengths due to downsampling in the encoder.


        embedding: A :py:class:`torch.nn.Module` which is an embedding lookup for
            targets (eg graphemes, wordpieces) which accepts a :py:`torch.Tensor`
            as input of size ``[batch, max_output_seq_len]``

        dec_rnn: A :py:class:`torch.nn.Module` containing the recurrent part of
            the RNN-T prediction.

            Must accept as input a tuple where the first element is the prediction
            network input (a :py:`torch.Tensor`) with size ``[batch, max_output_seq_len + 1,
            in_features]`` and the second element is a
            :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represent the sequence length of the corresponding *input*
            sequence to the rnn. *Note: this must be a `batch_first` rnn.*

            It must return a tuple where the first element is the result after
            applying the module to the input. It must have size
            ``[max_rnn_seq_len, batch, rnn_features]``. The second element of
            the tuple return value is a :py:class:`torch.Tensor` with size
            ``[batch]`` where each entry represents the sequence length of the
            corresponding *output* sequence. These may be different than the
            input sequence lengths due to downsampling in the encoder.

        fully_connected: A :py:class:`torch.nn.Module` containing the fully
            connected part of the RNNT joint network.

            Must accept as input a tuple where the first element is
            a :py:class:`torch.Tensor` with size ``[batch,
            encoder_out_seq_len, dec_rnn_out_seq_len, hidden_dim]`` where `hidden_dim`
            is the the concatenation of the `encoder` and `dec_rnn` outputs along
            the hidden dimension axis. The second element is and the
            input (a :py:class:`torch.Tensor`) with size ``[batch,
            max_fc_in_seq_len, max_fc_in_features]`` and the second element is
            a :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represents the sequence length of the corresponding *input*
            sequence to the fully connected layer(s). ``max_fc_in_seq_len`` and
            ``max_fc_in_features`` will either be ``max_rnn_seq_len`` and
            ``rnn_features`` or ``max_la_seq_len`` and ``la_features``
            depending on whether lookahead is :py:data:`None`.

            It must return a tuple where the first element is the result after
            applying the module to the previous layers output. It must have
            size ``[batch, max_out_seq_len, out_features]``. The second element
            is returned unchanged but in this context should be a Tuple of two
            :py:class:`torch.Tensor`s both of size ``[batch]`` that contain the *output* lengths of a) the audio features
            inputs and b) the target sequences.

    """

    def __init__(self, encoder, embedding, dec_rnn, fully_connected):
        super().__init__()

        assert (
            dec_rnn.batch_first == True
        ), "dec_rnn should be a batch_first rnn"
        self.encode = encoder
        self.embedding = embedding
        self.dec_rnn = dec_rnn
        self.fully_connected = fully_connected

        self.use_cuda = torch.cuda.is_available()

    def forward(
        self,
        x: Tuple[
            Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
        ],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Returns the result of applying the RNN-T network to the input audio
        features ``x[0][0]`` and target labels ``x[0][1]``.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: A Tuple ``(x[0], x[1])``. ``x[0]`` is input to the network and is
                a Tuple ``x[0] = (x[0][0], x[0][1])`` where both elements are
                :py:class:`torch.Tensor`s. ``x[0][0]`` is the audio feature input
                with size ``[max_seq_len, batch, in_features]`` while ``x[0][1]`` is
                the target label tensor of size ``[batch, max_label_length]``.
                ``x[1]`` is a Tuple of two :py:class:`torch.Tensor`s both of
                size ``[batch]`` that contain the *input* lengths of a) the audio feature
                inputs ``x[1][0]`` and b) the target sequences ``x[1][1]``.
        Returns:
            A Tuple where the first element is the output of the RNNT network: a
                :py:class:`torch.Tensor` with size ``[batch, max_seq_len,
                max_label_length + 1, vocab_size + 1]``, and the second element
                is a Tuple of two :py:class:`torch.Tensor`s both of
                size ``[batch]`` that contain the *output* lengths of a) the audio features
                inputs and b) the target sequences. These may be of different
                lengths than the inputs as a result of downsampling.

            """

        self._certify_inputs_forward(x)

        # Now use 'x_inp' to refer to audio features
        (x_inp, y), (x_lens, y_lens) = x

        if self.is_cuda:
            x_inp = x_inp.cuda()
            x_lens = x_lens.cuda()
            y = y.cuda()
            y_lens = y_lens.cuda()
        # TODO - convert to half precision here?

        X_ = (x_inp, x_lens)
        Y_ = (y, y_lens)

        f = self.encode(X_)  # f[0] = (T, B, H1)
        g = self.prediction(Y_)  # g[0] = (U, B, H2)

        f_inp = f[0]
        g_inp = g[0]
        seq_lengths = (f[1], g[1])  # (time_lens, label_lens)

        return self.joint(((f_inp, g_inp), seq_lengths))

    def prediction(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the result of applying the RNN-T prediction network to the
        target labels ``x[0][1]``. This function is only appropriate
        during training (when the ground-truth labels are available).
        TODO - reference where the inference function is found.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        Args:
            x: A Tuple where the first element is the target label tensor of
                size ``[batch, max_label_length]`` and the second is a
                :py:class:`torch.Tensor` of size ``[batch]`` that contain the
                *input* lengths of these target label sequences.

        Returns:
            Output from ``dec_rnn``. See initialisation docstring.
        """

        y_vals, y_lens = x

        y_vals = self.embedding(y_vals)
        y_vals = self._append_SOS(y_vals)
        return self.dec_rnn((y_vals, y_lens))

    def joint(
        self,
        x: Tuple[
            Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
        ],
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns the result of applying the RNN-T joint network to the encoder
        hidden state audio features ``x[0][0]`` and the prediction network
        target label hidden state ``x[0][1]``.

        Args:
            x: A Tuple ``(x[0], x[1])``. ``x[0][0]`` is the first element of the
                encoder network output (i.e. `.encode(...)[0]` - see
                :py:class:`.RNNTEncoder` docstring). ``x[0][1]`` is the first element of the
                prediction network output (i.e. `.prediction(...)[0]` - see prediction docstring).
                ``x[1]`` is a Tuple of two :py:class:`torch.Tensor`s both of
                size ``[batch]`` that contain the *input* lengths of a) the audio feature
                inputs ``x[1][0]`` and b) the target sequences ``x[1][1]``.
        Returns:
            The output of the :py:class:`.RNNT` network. See initialisation docstring.
        """
        (f, g), seq_lengths = x

        T, B1, H1 = f.shape
        B2, U_, H2 = g.shape

        assert (
            B1 == B2
        ), "Batch size from prediction network and transcription must be equal"

        f = f.transpose(1, 0)  # B1, T, H1
        f = f.unsqueeze(dim=2)  # (B, T, 1, H)
        f = f.expand((B1, T, U_, H1)).contiguous()

        g = g.unsqueeze(dim=1)  # (B, 1, U_, H)
        g = g.expand((B1, T, U_, H2)).contiguous()

        joint_inp = torch.cat([f, g], dim=3).contiguous()  # (B, T, U_, H1 + H2)

        return self.fully_connected((joint_inp, seq_lengths))

    def _append_SOS(self, y):
        """Appends the SOS token (all zeros) to the start of the target tensor"""

        B, U, H = y.shape
        # preprend blank
        start = torch.zeros((B, 1, H))
        y = torch.cat([start, y], dim=1)  # (B, U + 1, H)
        y = y.contiguous()
        return y

    def _certify_inputs_forward(self, inp):
        # inp = (x, x_lens), (y, y_lens) #TODO <- change from this
        inp = (x, y), (x_lens, y_lens)
        T, B1, I = x.shape
        B2, U = y.shape
        assert B1 == B2, "Batch size must be the same for inputs and targets"


class RNNTEncoder(torch.nn.Module):
    r"""`RNN-T <https://arxiv.org/pdf/1211.3711.pdf>`_ encoder (Transcription Network). Architecture
    based on `Streaming End-to-end Speech Recognition For Mobile Devices
    <https://arxiv.org/pdf/1811.06621.pdf>`_.

    Args:

        rnn1: A :py:class:`torch.nn.Module` containing the first recurrent part of
            the RNN-T encoder.

            Must accept as input a tuple where the first element is the network
            input (a :py:`torch.Tensor`) with size ``[max_seq_len, batch,
            in_features]`` and the second element is a
            :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represents the sequence length of the corresponding *input*
            sequence to the rnn.

            It must return a tuple where the first element is the result after
            applying the module to the input. It must have size
            ``[max_rnn_seq_len, batch, rnn_features]``. The second element of
            the tuple return value is a :py:class:`torch.Tensor` with size
            ``[batch]`` rnn2where each entry represents the sequence length of the
            corresponding *output* sequence. These may be different than the
            input sequence lengths due to downsampling in the encoder.

        time_reducer: An Optional ``Callable`` that reduces the number of timesteps
            into ``rnn2`` by stacking adjacent frames in the frequency dimension as
            employed in https://arxiv.org/pdf/1811.06621.pdf. This function takes two arguments:
            a Tuple of ``rnn1`` output and the sequence lengths of size ``[batch]``,
            and an optional ``time_reduction_factor`` (see below).
            This callable must return a Tuple where the first element is the result after
            applying the module to the input which must have size
            ``[ceil(max_rnn_seq_len/time_reduction_factor) , batch, rnn_features * time_reduction_factor ]``.
            The second element of the tuple must be a :py:class:`torch.Tensor` of size ``[batch]``
            that contains the new length of the corresponding sequence.

        time_reduction_factor: An Optional ``int`` with default value 2. If
            ``time_reducer`` is not None, this is the ratio by which the time dimension
            is reduced.

        rnn2: An Optional :py:class:`torch.nn.Module` containing the second
            recurrent part of the RNN-T encoder. This must be None unless ``time_reducer``
            is also passed.

            Must accept as input a tuple where the first element is the output
            from time_reducer (a :py:`torch.Tensor`) with size ``[max_downsampled_seq_len, batch,
            rnn1_features * time_reduction_factor]`` and the second element is a
            :py:class:`torch.Tensor` of size ``[batch]`` where each entry
            represents the sequence length of the corresponding *input*
            sequence to the rnn.

            It must return a tuple where the first element is the result after
            applying the module to the output. It must have size
            ``[max_downsampled_seq_len, batch, rnn_features]``. The second element of
            the tuple return value is a :py:class:`torch.Tensor` with size
            ``[batch]`` where each entry represents the sequence length of the
            corresponding *output* sequence. These may be different than the
            input sequence lengths due to downsampling.

    Returns:
        A Tuple where the first element is the  output of ``rnn2`` if it is not
        None, else the output of ``rnn1`` and the second element is a
        :py:class:`torch.Tensor` of size ``[batch]`` where each entry
        represents the sequence length of the corresponding *output*
        sequence to the encoder.

    """

    def __init__(
        self,
        rnn1: torch.nn.Module,
        time_reducer: Optional[torch.nn.Module] = None,
        time_reduction_factor: Optional[int] = 2,
        rnn2: Optional[torch.nn.Module] = None,
    ):
        assert isinstance(time_reduction_factor, int)
        if time_reducer is None:
            assert (
                rnn2 is None
            ), "Do not pass rnn2 without a time_reducer Callable"
            assert (
                time_reduction_factor == 1
            ), f"if `time_reducer` is None, must have time_reduction_factor == 1 but it is = {time_reduction_factor}"
        else:
            assert (
                time_reduction_factor > 1
            ), f"time_reduction_factor must be > 2 but = {time_reduction_factor}"

        assert rnn1.rnn.batch_first == False
        if rnn2:
            assert rnn2.rnn.batch_first == False

        super().__init__()

        self.rnn1 = rnn1
        self.time_reducer = time_reducer
        self.time_reduction_factor = time_reduction_factor
        self.rnn2 = rnn2

        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.rnn1 = self.rnn1.cuda()
            if self.rnn2 is not None:
                self.rnn2 = self.rnn2.cuda()

    def forward(
        self, x: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the result of applying the RNN-T encoder to the input audio features.

        All inputs are moved to the GPU with :py:meth:`torch.nn.Module.cuda` if
        :py:func:`torch.cuda.is_available` was :py:data:`True` on
        initialisation.

        See :py:class:`.RNNTEncoder` for detailed information about the input
        and output of each module.

        Args:
            x: Input for the first rnn module. See initialisation docstring.

        Returns:
            Output from ``rnn2`` if present, else output from ``rnn1``. See initialisation
            docstring.
        """

        if self.use_cuda:
            h = (x[0].cuda(), x[1].cuda())

        h = self.rnn1(h)

        if self.time_reducer:
            h = self.time_reducer(h, self.time_reduction_factor)

            h = self.rnn2(h)

        return (h[0].contiguous(), h[1])
