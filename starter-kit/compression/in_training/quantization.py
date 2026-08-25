"""
UdaciSense Project: Quantization-Aware Training Module

This module provides a quantizable MobileNetV3 model implementation for the household objects 
dataset, along with functions for quantization-aware training and model conversion.
"""

import copy
import time
from typing import Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.ao.quantization
import torchvision.models as tv_models
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
try:
    from torchvision.models.quantization.mobilenetv3 import _mobilenet_v3_conf, _mobilenet_v3_model
except Exception:  # pragma: no cover - version dependent fallback
    _mobilenet_v3_conf = None
    _mobilenet_v3_model = None
from tqdm import tqdm

from utils.model import get_model_size, save_model, train_single_epoch, validate_single_epoch


def _report_forced_float_modules(tag: str, forced_float: Tuple[str, ...]) -> None:
    """Print the modules intentionally kept in fp32 for numerical stability."""
    if not forced_float:
        return
    print(f"[{tag}] Forcing fp32 on {len(forced_float)} sensitive module(s):")
    for name in forced_float:
        print(f"  - {name}")


def _apply_mobilenetv3_safe_qconfig_overrides(model: nn.Module, backend: str):
    """Keep MobileNetV3's SE / hard-swish pathways in float while quantizing the rest of the trunk."""
    forced_float = []
    qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
    model.qconfig = qconfig

    for name, module in model.named_modules():
        if not hasattr(module, "qconfig"):
            continue
        name_l = name.lower()
        is_sensitive = (
            isinstance(module, (nn.Hardswish, nn.Sigmoid, nn.ReLU6))
            or "se" in name_l
            or "hardswish" in name_l
            or "sigmoid" in name_l
            or "relu6" in name_l
        )
        if is_sensitive:
            module.qconfig = None
            forced_float.append(name)

    _report_forced_float_modules("QAT", tuple(forced_float))
    return forced_float


# Batch-norm freezing utility (import path moved across torch versions).
try:  # pragma: no cover - import path is version dependent
    from torch.ao.nn.intrinsic.qat import freeze_bn_stats as _freeze_bn_stats
except Exception:  # pragma: no cover
    from torch.nn.intrinsic.qat import freeze_bn_stats as _freeze_bn_stats


def _rebuild_optimizer(
    optimizer: torch.optim.Optimizer, model: nn.Module
) -> torch.optim.Optimizer:
    """Recreate an optimizer of the same type over a model's current parameters.

    prepare_qat() fuses Conv+BN and inserts fake-quant modules, replacing some
    Parameters. The pre-existing optimizer still references the old Parameters,
    so we rebuild it over model.parameters() using the same hyperparameters.
    """
    return type(optimizer)(model.parameters(), **optimizer.defaults)


class QuantizableMobileNetV3_Household(nn.Module):
    """Quantizable MobileNetV3 model for household objects dataset.
    
    This model is designed to be compatible with PyTorch's quantization features,
    including quantization-aware training (QAT).
    
    Attributes:
        model: The underlying MobileNetV3 model with a modified classifier
    """
    
    def __init__(
        self, 
        num_classes: int = 10, 
        dropout_rate: float = 0.2, 
        quantize: bool = False, 
        pretrained: bool = True
    ):
        """Initialize a quantizable MobileNetV3 model.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout probability in the classifier
            quantize: Whether to create a quantization-ready model
            pretrained: Whether to load ImageNet pretrained weights
        """
        super().__init__()
        
        # Create a quantizable MobileNetV3 Small with a safe fallback for
        # torchvision versions that do not expose the private quantization
        # constructor names used by older implementations.
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        if _mobilenet_v3_conf is not None and _mobilenet_v3_model is not None:
            inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small")
            self.model = _mobilenet_v3_model(
                inverted_residual_setting=inverted_residual_setting,
                last_channel=last_channel,
                weights=weights,
                progress=True,
                quantize=quantize,
            )
        else:
            quant_ns = getattr(tv_models, "quantization", None)
            if quant_ns is not None and hasattr(quant_ns, "mobilenet_v3_small"):
                self.model = quant_ns.mobilenet_v3_small(
                    weights=weights,
                    progress=True,
                    quantize=quantize,
                )
            else:
                self.model = tv_models.mobilenet_v3_small(
                    weights=weights,
                    progress=True,
                )
        
        # Modify the classifier for the household objects dataset
        last_channel = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(last_channel, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Output tensor of shape [B, num_classes]
        """
        # Resize the image to the format expected by MobileNetV3
        x = torch.nn.functional.interpolate(
            x, size=(224, 224), mode='bilinear', align_corners=False
        )
        return self.model(x)
    
    def fuse_model(self, is_qat: bool = False) -> 'QuantizableMobileNetV3_Household':
        """Fuse operations like Conv+BN+ReLU for improved performance.
        
        Args:
            is_qat: Whether the fusion is for quantization-aware training
            
        Returns:
            Self with fused operations
        """
        # The torchvision quantizable MobileNetV3 backbone already knows which
        # Conv+BN(+activation) patterns to fuse; delegate to it. Fusion folds BN
        # into conv and merges activations, which both speeds up inference and
        # gives quantization a single, well-conditioned op to observe.
        self.model.fuse_model(is_qat=is_qat)
        return self


def _prepare_qat_model(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    """Prepare model for quantization-aware training.
    
    This function performs the necessary steps to convert a regular model
    to be ready for quantization-aware training.
    
    Args:
        model: Model to prepare for QAT
        backend: Quantization backend to use ("fbgemm" or "qnnpack")
    
    Returns:
        Model prepared for QAT
    """
    # 1) Select the backend kernels (fbgemm=x86, qnnpack=ARM/mobile).
    torch.backends.quantized.engine = backend

    # 2) Fuse modules in train mode (QAT fusion keeps BN as a trainable folded op).
    model.train()
    if hasattr(model, "fuse_model"):
        model.fuse_model(is_qat=True)

    # 3) Attach the default QAT qconfig but keep the numerically fragile SE /
    #    hard-swish branches in fp32 so they do not destabilize training.
    #    IMPORTANT: do not overwrite the root qconfig after the per-module override,
    #    or the quantization policy resets back to the default and the fragile
    #    MobileNetV3 blocks are quantized again.
    _apply_mobilenetv3_safe_qconfig_overrides(model, backend)

    # 4) Insert fake-quant / observer modules in place so the network learns to
    #    be robust to int8 rounding during the remaining training epochs.
    torch.ao.quantization.prepare_qat(model, inplace=True)
    return model


def _convert_qat_model_to_quantized(model: nn.Module) -> nn.Module:
    """Convert a QAT model to a fully quantized model for inference.
    
    Args:
        model: QAT-trained model
        
    Returns:
        Fully quantized model
    """
    # Conversion runs on CPU in eval mode: observers are removed and weights /
    # activations are materialized as int8, producing the deployable model.
    model.eval()
    model_cpu = model.to("cpu")
    quantized_model = torch.ao.quantization.convert(model_cpu, inplace=False)
    return quantized_model


def train_model_qat(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    training_config: Dict[str, Any],
    checkpoint_path: str,
    backend: str = "fbgemm",
) -> Tuple[nn.Module, Dict[str, Any], float, int]:
    """Train a model using quantization-aware training.
    
    This function implements the complete QAT workflow, including:
    1. Initial training before QAT
    2. QAT activation and fine-tuning
    3. Observer disabling and batch norm freezing
    4. Final conversion to a fully quantized model
    
    Args:
        model: PyTorch model (should support fuse_model method)
        train_loader: Training data loader
        test_loader: Test data loader
        training_config: Dictionary containing training configuration
        checkpoint_path: Path to save the best QAT model
        backend: Quantization backend ("fbgemm" for x86, "qnnpack" for ARM)
        
    Returns:
        Tuple of (quantized_model, training_stats, best_accuracy, best_epoch)
    """
    # Step 1: Define training variables
    
    # Extract training configuration
    num_epochs = training_config.get('num_epochs', 100)
    criterion = training_config.get('criterion')
    optimizer = training_config.get('optimizer')
    scheduler = training_config.get('scheduler')
    patience = training_config.get('patience', 10)
    device = training_config.get('device', 
                                torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    grad_clip_norm = training_config.get('grad_clip_norm', None)
    freeze_bn_epochs = training_config.get('freeze_bn_epochs', 0)  # Default: don't freeze BN
    qat_start_epoch = training_config.get('qat_start_epoch', 0)  # When to start QAT
    scheduler = training_config.get('scheduler')
    # Epoch at which to freeze activation observers so int8 scales/zero-points
    # stop moving and stabilize before final convergence.
    disable_observer_epoch = training_config.get(
        'disable_observer_epoch', qat_start_epoch + 2
    )
    
    print(f"Training with quantization-aware training for {num_epochs} epochs")
    print(f"QAT start epoch: {qat_start_epoch}, Finetune BN stats epochs: {freeze_bn_epochs}")
    print(f"QAT will be activated after epoch {qat_start_epoch}")
        
    # Training statistics
    best_accuracy = 0.0
    best_epoch = 0
    training_stats = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "epoch_time": [],
        "lr": []
    }
    
    # Early stopping variable
    early_stop_counter = 0   
    
    # Step 2: Train the model with QAT
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Make sure model is in train mode
        model.train()
        
        # Prepare model for QAT at the start of the QAT epoch. Fusing + inserting
        # fake-quant modules replaces some parameters, so we rebuild the optimizer
        # over the new parameters and re-point the scheduler at it.
        if epoch == qat_start_epoch:
            print("Preparing model for quantization-aware training...")
            model = _prepare_qat_model(model, backend=backend)
            model.to(device)
            optimizer = _rebuild_optimizer(optimizer, model)
            if scheduler is not None and hasattr(scheduler, "optimizer"):
                scheduler.optimizer = optimizer
        
        # Train for one epoch
        train_loss, train_accuracy = train_single_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip_norm=grad_clip_norm, epoch=epoch, num_epochs=num_epochs,
        )
        
        # Disable observers after sufficient QAT training to freeze the learned
        # quantization parameters (scales / zero-points) for stable inference.
        if epoch >= disable_observer_epoch:
            model.apply(torch.ao.quantization.disable_observer)
        
        # Freeze batch-norm running statistics once we reach freeze_bn_epochs so
        # BN behaves consistently between training and the converted int8 model.
        if freeze_bn_epochs and epoch >= qat_start_epoch and epoch >= freeze_bn_epochs:
            model.apply(_freeze_bn_stats)

        # Evaluate on test set
        if epoch >= qat_start_epoch:
            # IMPORTANT! Move model to CPU for inference
            eval_model = copy.deepcopy(model).cpu()
            eval_model.eval()
            
            # Convert the current QAT model to a real int8 model for evaluation
            # so reported accuracy reflects true quantized inference.
            quantized_model = _convert_qat_model_to_quantized(eval_model)
            
            # Evaluate quantized model
            test_loss, test_accuracy = validate_single_epoch(
                quantized_model, test_loader, criterion, torch.device("cpu"), epoch, num_epochs
            )

            # Release the temporary eval copies promptly to avoid memory buildup.
            del eval_model, quantized_model
        else:
            # Evaluate fp32 model
            test_loss, test_accuracy = validate_single_epoch(
                model, test_loader, criterion, device, epoch, num_epochs
            )
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()
        
        # Record epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Print statistics
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, "
              f"LR: {lr:.6f}, Time: {epoch_time:.2f}s")
        
        # Save best model
        if test_accuracy > best_accuracy and epoch >= qat_start_epoch:
            print(f"New best quantized model! Saving... ({test_accuracy:.2f}%)")
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            
            save_model(model, checkpoint_path)
            early_stop_counter = 0  # Reset early stopping counter
        else:
            early_stop_counter += 1
        
        # Early stopping condition
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. No improvement for {patience} epochs.")
            break
        
        # Record statistics
        training_stats["epoch"].append(epoch + 1)
        training_stats["train_loss"].append(train_loss)
        training_stats["train_accuracy"].append(train_accuracy)
        training_stats["test_loss"].append(test_loss)
        training_stats["test_accuracy"].append(test_accuracy)
        training_stats["epoch_time"].append(epoch_time)
        training_stats["lr"].append(lr)
    
    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    print(f"Best QAT model saved as '{checkpoint_path}' at epoch {best_epoch}")
    
    # Step 3: Convert the best QAT model to final quantized model for inference
    print("Converting best QAT model to fully quantized model...")
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    quantized_model = _convert_qat_model_to_quantized(model)
    
    return quantized_model, training_stats, best_accuracy, best_epoch