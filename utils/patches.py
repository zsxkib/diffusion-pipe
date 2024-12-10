from typing import Optional

import torch
from torch import nn
import peft
from peft.tuners._buffer_dict import BufferDict


def _move_adapter_to_device_of_base_layer(self, adapter_name: str, device: Optional[torch.device] = None) -> None:
    """
    Move the adapter of the given name to the device of the base layer.
    """
    if device is None:
        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                device = weight.device
                dtype = weight.dtype
                break
        else:
            # no break encountered: could not determine the device
            return

    meta = torch.device("meta")

    # loop through all potential adapter layers and move them to the device of the base layer; be careful to only
    # move this specific adapter to the device, as the other adapters could be on different devices
    # see #1639
    for adapter_layer_name in self.adapter_layer_names + self.other_param_names:
        adapter_layer = getattr(self, adapter_layer_name, None)
        if not isinstance(adapter_layer, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
            continue
        if adapter_name not in adapter_layer:
            continue
        if any(p.device == meta for p in adapter_layer.parameters()):
            continue

        if ((weight.dtype.is_floating_point or weight.dtype.is_complex)
            # This is the part I added.
            and not (weight.dtype == torch.float8_e4m3fn or weight.dtype == torch.float8_e5m2)):
            adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device, dtype=dtype)
        else:
            adapter_layer[adapter_name] = adapter_layer[adapter_name].to(device)


def apply_patches():
    # Prevent PEFT from downcasting LoRA weights to fp8 only for this script to upcast them again.
    # TODO: probably should send a PR to PEFT. Default behavior looks like a mistake to me.
    peft.tuners.tuners_utils.BaseTunerLayer._move_adapter_to_device_of_base_layer = _move_adapter_to_device_of_base_layer
