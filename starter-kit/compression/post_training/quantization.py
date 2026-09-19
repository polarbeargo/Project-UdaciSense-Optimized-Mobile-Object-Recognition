"""
UdaciSense Project: Post-Training Quantization Module

This module provides utilities for applying post-training quantization to PyTorch models,
supporting both static and dynamic quantization methods.
"""

import os
import copy
import gc
import operator
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.ao.quantization import QConfigMapping, get_default_qconfig_mapping
try:
    # Preferred on newer PyTorch: x86-tuned FX backend config.
    from torch.ao.quantization.backend_config import get_x86_backend_config
except ImportError:  # pragma: no cover - depends on installed torch version
    # Older/other builds only expose the native backend config, which is the
    # equivalent stable fallback for x86 CPUs (fbgemm).
    from torch.ao.quantization.backend_config import (
        get_native_backend_config as get_x86_backend_config,
    )
try:
    # Preferred on newer PyTorch: qnnpack-tuned FX backend config (ARM).
    from torch.ao.quantization.backend_config import get_qnnpack_backend_config
except ImportError:  # pragma: no cover - depends on installed torch version
    # Older builds only expose the native backend config; it is the stable
    # fallback for ARM too (qnnpack kernels are selected via the engine).
    from torch.ao.quantization.backend_config import (
        get_native_backend_config as get_qnnpack_backend_config,
    )
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.utils.data import DataLoader
from tqdm import tqdm


def _extract_tensor_input(batch: Any) -> Optional[torch.Tensor]:
    """Pull a representative input tensor out of an arbitrary dataloader batch.

    Pure and side-effect free (no captured state, no tensor mutation), so it is
    defined once at module scope instead of being rebuilt on every calibration
    call. Supports plain tensors, ``(inputs, labels)`` tuples/lists, and dict
    batches keyed by common input names.
    """
    if isinstance(batch, torch.Tensor):
        return batch
    if isinstance(batch, dict):
        for key in ("images", "image", "inputs", "input", "x", "data"):
            if key in batch:
                found = _extract_tensor_input(batch[key])
                if isinstance(found, torch.Tensor):
                    return found
        for value in batch.values():
            found = _extract_tensor_input(value)
            if isinstance(found, torch.Tensor):
                return found
        return None
    if isinstance(batch, (list, tuple)):
        for item in batch:
            found = _extract_tensor_input(item)
            if isinstance(found, torch.Tensor):
                return found
        return None
    return None


def _get_fx_backend_config(backend: str):
    """Return the FX backend_config that matches the target CPU ISA.

    The backend_config decides which op patterns are quantizable, so it must
    match the runtime engine: qnnpack on ARM, x86/fbgemm otherwise. Using the
    x86 config while the engine is qnnpack (or vice versa) can select the wrong
    kernels and hurt correctness/latency.
    """
    if backend == "qnnpack":
        return get_qnnpack_backend_config()
    return get_x86_backend_config()


def _get_mobilenetv3_safe_qconfig_mapping(backend: str) -> QConfigMapping:
    """Build an FX qconfig mapping that keeps MobileNetV3's fragile ops in fp32.

    Torchvision's plain MobileNetV3 uses a `SqueezeExcitation` whose
    `scale * input` and the hard-swish / hard-sigmoid activations do not have
    quantized kernels for this eager path, which raises
    `empty_strided not supported on quantized tensors`. FX graph-mode PTQ lets us
    globally quantize the conv/linear trunk while pinning those sensitive nodes
    (SE blocks, activations, elementwise add/mul, classifier, stem) back to fp32.
    """
    # `get_default_qconfig_mapping` already returns a fresh QConfigMapping for
    # the requested backend, so we mutate it directly (older torch builds do
    # not expose the QConfigMapping.from_compat_dict round-trip helper).
    qconfig_mapping = get_default_qconfig_mapping(backend)

    # Keep residual/SE elementwise math in fp32 (these are the ops that raise
    # the quantized empty_strided error).
    for op in (operator.add, operator.iadd, operator.mul, operator.imul,
               torch.add, torch.mul):
        qconfig_mapping.set_object_type(op, None)

    # Keep hard-swish / hard-sigmoid / sigmoid activations in fp32.
    sensitive_ops = [
        F.hardswish, F.hardsigmoid, F.sigmoid, F.relu6, F.adaptive_avg_pool2d,
        torch.sigmoid, torch.flatten,
        nn.Hardswish, nn.Hardsigmoid, nn.Sigmoid, nn.ReLU6, nn.AdaptiveAvgPool2d,
    ]
    # Some functional aliases (e.g. torch.hardswish / torch.hardsigmoid) are not
    # present on every torch build, so only add them when they actually exist.
    for _opt_name in ("hardswish", "hardsigmoid"):
        _opt_op = getattr(torch, _opt_name, None)
        if callable(_opt_op):
            sensitive_ops.append(_opt_op)
    for op in sensitive_ops:
        qconfig_mapping.set_object_type(op, None)

    # Keep the SE subgraphs, activation bridges, classifier head and stem in fp32.
    for pattern in (r".*se.*", r".*SqueezeExcitation.*", r".*activation.*",
                    r".*classifier.*"):
        qconfig_mapping.set_module_name_regex(pattern, None)
    for name in ("model.classifier", "model.features.0", "model.features.0.0",
                 "classifier", "features.0", "features.0.0"):
        qconfig_mapping.set_module_name(name, None)

    return qconfig_mapping


class QuantizableMobileNetV3_Household(nn.Module):
    """
    Wraps a MobileNetV3_Household model with QuantStub/DeQuantStub so that the
    model can go through PyTorch's eager-mode static quantization workflow
    (prepare -> calibrate -> convert).

    The wrapped model's activations are quantized right after the input
    (via `quant`) and de-quantized right before the output (via `dequant`);
    everything in between runs through the original model's layers, which
    get swapped for their quantized counterparts during `convert`.
    """

    def __init__(self, original_model):
        super().__init__()
        self.quant = QuantStub()
        self.model = original_model
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

    def fuse_model(self) -> None:
        """Fuse Conv+BN(+ReLU) patterns in place for better quantization.

        Folds BatchNorm into the preceding Conv (and an optional trailing
        ReLU/ReLU6) so int8 inference is faster and more accurate. Only the
        eager-fusable Conv+BN, Conv+BN+ReLU and Conv+ReLU patterns are matched;
        MobileNetV3's Hardswish/Hardsigmoid activations are intentionally left
        unfused (no eager fusion kernel exists for them).
        """
        print("Fusing layers...")

        # BN folding is only valid in eval mode; remember and restore the flag.
        was_training = self.model.training
        self.model.eval()

        fusable = (nn.ReLU, nn.ReLU6)
        modules_to_fuse: List[List[str]] = []

        try:
            # Fusion is deferred until after this loop, so the module tree is
            # never mutated mid-iteration -- iterate the generator lazily
            # instead of materializing every (name, module) pair up front.
            for name, module in self.model.named_modules():
                if not isinstance(module, nn.Sequential):
                    continue

                children = list(module.named_children())
                prefix = f"{name}." if name else ""
                i, n = 0, len(children)
                while i < n:
                    _, current = children[i]
                    if not isinstance(current, nn.Conv2d):
                        i += 1
                        continue

                    group = [prefix + children[i][0]]
                    j = i + 1
                    if j < n and isinstance(children[j][1], nn.BatchNorm2d):
                        group.append(prefix + children[j][0])
                        j += 1
                        if j < n and isinstance(children[j][1], fusable):
                            group.append(prefix + children[j][0])
                            j += 1

                    if len(group) >= 2:
                        modules_to_fuse.append(group)
                    i = j

            if modules_to_fuse:
                print(f"Found {len(modules_to_fuse)} fusable pattern(s).")
                tq.fuse_modules(self.model, modules_to_fuse, inplace=True)
            else:
                print("No fusable Conv-BN(-ReLU) patterns found; skipping fusion.")
        finally:
            # Restore the caller's original mode even if fusion raises.
            self.model.train(was_training)


def quantize_model(
    model: nn.Module,
    calibration_data_loader: Optional[DataLoader] = None,
    calibration_num_batches: Optional[int] = None,
    quantization_type: str = "dynamic",
    backend: str = "x86",
) -> nn.Module:
    """Apply post-training quantization to a PyTorch model.
    
    Args:
        model: The original model to quantize
        calibration_data_loader: DataLoader for calibration data,
            required for static quantization
        calibration_num_batches: Number of batches to run calibration on
        quantization_type: Type of quantization to apply:
            - "dynamic": Dynamic quantization (weights are quantized, activations quantized during inference)
            - "static": Static quantization (weights and activations are pre-quantized)
        backend: Quantization backend, either "fbgemm" (x86) or "qnnpack" (ARM)
            
    Returns:
        Quantized model
        
    Raises:
        ValueError: If an unsupported backend or quantization type is specified,
                   or if static quantization is requested without calibration data
    """
    # Verify backend
    if backend not in ["x86","fbgemm", "qnnpack"]:
        raise ValueError("Backend must be either 'fbgemm' (x86) or 'qnnpack' (ARM)")

    # Create a copy of the model for quantization
    model_to_quantize = copy.deepcopy(model)
    
    # Set model to evaluation mode
    model_to_quantize.eval()
    
    # NOTE: Feel free to not implement all quantization types
    # Apply quantization based on type
    if quantization_type.lower() == "dynamic":
        return _apply_dynamic_quantization(model_to_quantize)
    elif quantization_type.lower() == "static":
        if calibration_data_loader is None:
            raise ValueError("Static quantization requires a calibration_data_loader")
        return _apply_static_quantization(model_to_quantize, calibration_data_loader, calibration_num_batches, backend)
    else:
        raise ValueError(f"Unsupported quantization type: {quantization_type}")

# TODO: Implement dynamic quantization, if selected
# Remember to look at built-in pytorch functionalities whenever possible
def _apply_dynamic_quantization(
    model: nn.Module
) -> nn.Module:
    """Apply dynamic quantization to a model.
    
    Dynamic quantization quantizes weights ahead of time but quantizes activations
    dynamically during inference.
    
    Args:
        model: Model to quantize (in eval mode)
        
    Returns:
        Dynamically quantized model
    """
    print("Applying dynamic quantization...")

    # Dynamic quantization targets weight-heavy layers (Linear/RNN) and keeps
    # activations in fp32 until inference. This shrinks the model (int8 weights)
    # and speeds up CPU matmuls with essentially no calibration required.
    #
    # `model` is already an owned deep copy made in `quantize_model`, so convert
    # it IN PLACE: this avoids a second full-model deep copy (`inplace=False`
    # is the default) and the transient peak memory / garbage it produces.
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
        inplace=True,
    )
    return quantized_model
                

# TODO: Implement static quantization, if selected
# Remember to look at built-in pytorch functionalities whenever possible
# And that you first need to prepare the model for quantization, then apply calibration, and finally convert the model to quantized
def _apply_static_quantization(
    model: nn.Module,
    calibration_data_loader: DataLoader,
    calibration_num_batches: Optional[int] = None,
    backend: str = "x86",
) -> nn.Module:
    """Apply static quantization to a model using provided calibration data.
    
    Static quantization quantizes both weights and activations ahead of time.
    This uses FX graph-mode PTQ so that MobileNetV3's SqueezeExcitation and
    hard-swish sub-graphs can be pinned to fp32 (they have no quantized kernels
    on this path and otherwise raise "empty_strided not supported on quantized
    tensors"), while the conv/linear trunk is quantized to int8.
    
    Args:
        model: Model to quantize (in eval mode)
        calibration_data_loader: DataLoader for calibration data
        calibration_num_batches: Number of batches to use for calibration
        backend: Quantization backend, either "fbgemm" (x86) or "qnnpack" (ARM)
        
    Returns:
        Statically quantized model
    """
    print("Applying static quantization...")
    device = next(model.parameters()).device
    # If calibration_num_batches is not specified, use all available batches
    if calibration_num_batches is None:
        calibration_num_batches = len(calibration_data_loader)

    # Make sure the requested backend is actually used for quantized kernels.
    # (Process-global; int8 inference on the returned model reads it too.)
    torch.backends.quantized.engine = backend

    model.eval()

    # FX graph-mode PTQ: a representative example input is needed to trace.
    first_batch = next(iter(calibration_data_loader))
    inputs_sample = _extract_tensor_input(first_batch)
    if not isinstance(inputs_sample, torch.Tensor):
        raise ValueError("Could not extract a valid tensor from the data loader batch.")
    example_inputs = (inputs_sample.to(device),)

    qconfig_mapping = _get_mobilenetv3_safe_qconfig_mapping(backend)
    backend_config = _get_fx_backend_config(backend)

    prepared_model = quantize_fx.prepare_fx(
        model,
        qconfig_mapping,
        example_inputs,
        backend_config=backend_config,
    )

    # Calibrate: run representative data through the model so the inserted
    # observers can record activation statistics (min/max ranges).
    print(f"Calibrating with up to {calibration_num_batches} batch(es)...")
    prepared_model.eval()
    with torch.inference_mode():
        for i, batch in enumerate(tqdm(calibration_data_loader, total=calibration_num_batches)):
            if i >= calibration_num_batches:
                break
            inputs = _extract_tensor_input(batch)
            if not isinstance(inputs, torch.Tensor):
                raise ValueError("Calibration batch must contain tensor inputs.")
            prepared_model(inputs.to(device))

    # Convert the observed model into a truly quantized int8 model.
    print("Convert model")
    quantized_model = quantize_fx.convert_fx(prepared_model, backend_config=backend_config)

    # Drop the observer-laden prepared graph promptly so its buffers are not
    # kept alive by this frame after conversion (avoids a transient memory leak
    # of per-node observers).
    del prepared_model
    gc.collect()

    return quantized_model