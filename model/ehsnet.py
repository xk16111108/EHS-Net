from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Tuple, Union

import torch
from torch import nn

from .architecture import EHSNetArchitecture


CheckpointPath = Union[str, Path]
StateDict = MutableMapping[str, torch.Tensor]


class EHSNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 1,
        depths: Tuple[int, ...] = (2, 2, 2, 2),
        depths_decoder: Tuple[int, ...] = (2, 2, 2, 1),
        dims: Tuple[int, ...] = (96, 192, 384, 768),
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.network = EHSNetArchitecture(
            in_chans=input_channels,
            num_classes=num_classes,
            depths=depths,
            depths_decoder=depths_decoder,
            dims=dims,
            drop_path_rate=drop_path_rate,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(
                f"Expected a 4D input tensor, but received shape {x.shape}."
            )
        if x.size(1) == 1 and self.input_channels == 3:
            x = x.repeat(1, 3, 1, 1)
        if x.size(1) != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, but received "
                f"{x.size(1)}."
            )
        return self.network(x)

    def load_weights(
        self,
        checkpoint_path: CheckpointPath,
        map_location: Any = "cpu",
        strict: bool = True,
    ) -> Any:
        return load_checkpoint(
            self,
            checkpoint_path,
            map_location=map_location,
            strict=strict,
        )


def _is_state_dict(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and bool(value)
        and all(
            isinstance(key, str) and torch.is_tensor(tensor)
            for key, tensor in value.items()
        )
    )


def _load_checkpoint_file(
    checkpoint_path: CheckpointPath,
    map_location: Any,
) -> Any:
    try:
        return torch.load(
            str(checkpoint_path),
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(str(checkpoint_path), map_location=map_location)


def _extract_state_dict(checkpoint: Any) -> StateDict:
    if _is_state_dict(checkpoint):
        return OrderedDict(checkpoint)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("The checkpoint is neither a state dictionary nor a mapping.")
    for key in ("model", "state_dict", "model_state_dict", "net", "network"):
        candidate = checkpoint.get(key)
        if _is_state_dict(candidate):
            return OrderedDict(candidate)
    raise KeyError(
        "No state dictionary was found. Supported checkpoint keys are "
        "model, state_dict, model_state_dict, net, and network."
    )


def _strip_common_prefix(state_dict: StateDict, prefix: str) -> StateDict:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return OrderedDict(
            (key[len(prefix) :], value) for key, value in state_dict.items()
        )
    return state_dict


def _normalize_state_dict(state_dict: StateDict) -> StateDict:
    normalized = state_dict
    for prefix in ("module.", "model."):
        normalized = _strip_common_prefix(normalized, prefix)

    if normalized and all(key.startswith("vmunet.") for key in normalized):
        return OrderedDict(
            (f"network.{key[len('vmunet.'):]}", value)
            for key, value in normalized.items()
        )
    if normalized and all(key.startswith("network.") for key in normalized):
        return normalized
    return OrderedDict((f"network.{key}", value) for key, value in normalized.items())


def _load_state_dict(
    model: EHSNet,
    state_dict: StateDict,
    strict: bool,
) -> Any:
    normalized = _normalize_state_dict(state_dict)
    return model.load_state_dict(normalized, strict=strict)


def load_checkpoint(
    model: EHSNet,
    checkpoint_path: CheckpointPath,
    map_location: Any = "cpu",
    strict: bool = True,
) -> Any:
    checkpoint = _load_checkpoint_file(checkpoint_path, map_location)
    state_dict = _extract_state_dict(checkpoint)
    return _load_state_dict(model, state_dict, strict)


def build_model_from_checkpoint(
    checkpoint_path: CheckpointPath,
    map_location: Any = "cpu",
    strict: bool = True,
    **model_kwargs: Any,
) -> EHSNet:
    checkpoint = _load_checkpoint_file(checkpoint_path, map_location)
    state_dict = _extract_state_dict(checkpoint)
    output_weights = [
        tensor
        for key, tensor in state_dict.items()
        if key.endswith("final_conv.weight")
    ]
    if len(output_weights) != 1:
        raise KeyError("Unable to infer num_classes from final_conv.weight.")

    inferred_classes = int(output_weights[0].shape[0])
    requested_classes = model_kwargs.setdefault("num_classes", inferred_classes)
    if requested_classes != inferred_classes:
        raise ValueError(
            f"num_classes={requested_classes} does not match the checkpoint "
            f"output channels ({inferred_classes})."
        )

    model = EHSNet(**model_kwargs)
    _load_state_dict(model, state_dict, strict)
    return model
