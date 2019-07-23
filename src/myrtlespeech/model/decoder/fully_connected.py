from typing import Optional, Tuple, Union

import torch

from myrtlespeech.model.seq_len_wrapper import SeqLenWrapper


class FullyConnected(torch.nn.Module):
    r"""A fully connected neural network.

    Args:
        in_features: Size of each input sample.

        out_features: Size of each output sample.

        num_hidden_layers: Number of hidden layers. Must be a non-negative
            integer. ``0`` means no hidden layers making this module the same
            as a single :py:class:`torch.nn.Linear` layer.

        hidden_size: The number of features output by each hidden layer, if
            any.

        hidden_activation_fn: The activation function applied after each hidden
            layer, if any.

        seq_len_support: TODO

    Attributes:
        fully_connected (Union[:py:class:`torch.nn.Linear`, :py:class:`torch.nn.Sequential`]):
            A :py:class:`torch.nn.Module` that implements the network specified
            by the class arguments. It is be an instance of
            :py:class:`torch.nn.Linear` if ``num_hidden_layers == 0`` otherwise
            it is an instance of :py:class:`torch.nn.Sequential`.

    Raises:
        :py:class:`ValueError`: If ``num_hidden_layers < 0``.

        :py:class:`ValueError`: If ``num_hidden_layers == 0 and hidden_size is not None``.

        :py:class:`ValueError`: If ``num_hidden_layers == 0 and hidden_activation_fn is not None``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_hidden_layers: int,
        hidden_size: Optional[int],
        hidden_activation_fn: Optional[torch.nn.Module],
        seq_len_support: bool = False,
    ):
        if num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be >= 0")

        if num_hidden_layers == 0:
            if hidden_size is not None:
                raise ValueError(
                    "num_hidden_layers==0 but hidden_size is not None"
                )
            if hidden_activation_fn is not None:
                raise ValueError(
                    "num_hidden_layers==0 but hidden_activation_fn is not None"
                )

        super().__init__()

        self.fully_connected = self._build_fully_connected(
            in_features,
            out_features,
            num_hidden_layers,
            hidden_size,
            hidden_activation_fn,
            seq_len_support,
        )

    def _build_fully_connected(
        self,
        in_features: int,
        out_features: int,
        num_hidden_layers: int,
        hidden_size: Optional[int],
        hidden_activation_fn: Optional[torch.nn.Module],
        seq_len_support: bool,
    ) -> Union[torch.nn.Linear, torch.nn.Sequential]:
        hidden_layers = []
        for _ in range(num_hidden_layers):
            hidden_layers.append(torch.nn.Linear(in_features, hidden_size))
            if hidden_activation_fn:
                hidden_layers.append(hidden_activation_fn)
            assert hidden_size is not None
            in_features = hidden_size

        module = torch.nn.Linear(in_features, out_features)

        if hidden_layers:
            module = torch.nn.Sequential(*hidden_layers, module)

        if not seq_len_support:
            return module

        return SeqLenWrapper(
            module=module, seq_lens_fn=lambda seq_lens: seq_lens
        )

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Returns the result of applying ``fully_connected`` to ``x``.

        Args:
            x: :py:class:`torch.Tensor` with shape ``[batch, *, in_features]``
               where ``*`` means any number of additional dimensions.


        Returns:
            A :py:class:`torch.Tensor` with shape ``[batch, *, out_features]``
            where ``*`` means any number of additional dimensions.
        """
        return self.fully_connected(x)
