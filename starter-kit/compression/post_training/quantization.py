"""
UdaciSense Project: Post-Training Quantization Module

This module provides utilities for applying post-training quantization to PyTorch models,
supporting both static and dynamic quantization methods.
"""

import os
import copy
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.ao.quantization.quantize_fx as quantize_fx
from torch.ao.quantization import get_default_qconfig_mapping
from torch.utils.data import DataLoader
from tqdm import tqdm


# Make MobileNetV3_Household model quantizable using quant/dequant stubs.
# This eager-mode wrapper is provided for completeness; the main quantize_model()
# path below uses FX graph-mode quantization, which fuses and quantizes
# automatically and is more reliable for the MobileNetV3 architecture.
class QuantizableMobileNetV3_Household(nn.Module):
    """Wrap an existing MobileNetV3_Household model with quant/dequant stubs."""

    def __init__(self, original_model: nn.Module):
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.model = copy.deepcopy(original_model)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

    def fuse_model(self) -> None:
        """
        Fuse Conv+BN and Conv+BN+ReLU layers for better quantization results.

        Fusing folds batch-norm into the preceding convolution and merges the
        activation, reducing the number of quantize/dequantize boundaries and
        improving both accuracy and inference speed of the quantized model.
        """
        print("Fusing layers...")

        # Collect Conv-BN(-ReLU) fusion patterns by scanning sequential blocks.
        for module in self.modules():
            if isinstance(module, nn.Sequential):
                names = [name for name, _ in module.named_children()]
                children = [child for _, child in module.named_children()]
                idx = 0
                to_fuse: List[List[str]] = []
                while idx < len(children):
                    if (
                        idx + 2 < len(children)
                        and isinstance(children[idx], nn.Conv2d)
                        and isinstance(children[idx + 1], nn.BatchNorm2d)
                        and isinstance(children[idx + 2], nn.ReLU)
                    ):
                        to_fuse.append([names[idx], names[idx + 1], names[idx + 2]])
                        idx += 3
                    elif (
                        idx + 1 < len(children)
                        and isinstance(children[idx], nn.Conv2d)
                        and isinstance(children[idx + 1], nn.BatchNorm2d)
                    ):
                        to_fuse.append([names[idx], names[idx + 1]])
                        idx += 2
                    else:
                        idx += 1
                if to_fuse:
                    torch.ao.quantization.fuse_modules(
                        module, to_fuse, inplace=True
                    )
        

def quantize_model(
    model: nn.Module,
    calibration_data_loader: Optional[DataLoader] = None,
    calibration_num_batches: Optional[int] = None,
    quantization_type: str = "dynamic",
    backend: str = "fbgemm",
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
    if backend not in ["fbgemm", "qnnpack"]:
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
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM, nn.GRU, nn.RNN, nn.LSTMCell, nn.GRUCell, nn.RNNCell},
        dtype=torch.qint8,
    )
    return quantized_model
                

# TODO: Implement static quantization, if selected
# Remember to look at built-in pytorch functionalities whenever possible
# And that you first need to prepare the model for quantization, then apply calibration, and finally convert the model to quantized
def _apply_static_quantization(
    model: nn.Module,
    calibration_data_loader: DataLoader,
    calibration_num_batches: Optional[int] = None,
    backend: str = "fbgemm",
) -> nn.Module:
    """Apply static quantization to a model using provided calibration data.
    
    Static quantization quantizes both weights and activations ahead of time.
    
    Args:
        model: Model to quantize (in eval mode)
        calibration_data_loader: DataLoader for calibration data
        calibration_num_batches: Number of batches to use for calibration
        backend: Quantization backend, either "fbgemm" (x86) or "qnnpack" (ARM)
        
    Returns:
        Statically quantized model
    """
    print("Applying static quantization...")
    
    # If calibration_num_batches is not specified, use all available batches
    if calibration_num_batches is None:
        calibration_num_batches = len(calibration_data_loader)

    # Static quantization pre-computes activation ranges from representative
    # data, so both weights AND activations run in int8 at inference time.
    # We use FX graph-mode quantization which automatically traces the model,
    # fuses Conv+BN+ReLU patterns, inserts observers, and converts to int8.
    torch.backends.quantized.engine = backend
    qconfig_mapping = get_default_qconfig_mapping(backend)

    # A representative example input is required to symbolically trace the model.
    example_inputs, _ = next(iter(calibration_data_loader))
    example_inputs = example_inputs[:1]

    model.eval()
    prepared_model = quantize_fx.prepare_fx(
        model, qconfig_mapping, example_inputs
    )

    # Calibration: run a few batches through the prepared model so the observers
    # can record realistic activation statistics. No gradients are needed.
    print(f"Calibrating on {calibration_num_batches} batch(es)...")
    with torch.inference_mode():
        for batch_idx, (inputs, _) in enumerate(
            tqdm(calibration_data_loader, total=calibration_num_batches, desc="Calibration")
        ):
            if batch_idx >= calibration_num_batches:
                break
            prepared_model(inputs)

    # Convert the calibrated model to a fully quantized int8 model.
    quantized_model = quantize_fx.convert_fx(prepared_model)
    return quantized_model