"""
UdaciSense Project: Quantization-Aware Training Module

This module provides a quantizable MobileNetV3 model implementation for the household objects 
dataset, along with functions for quantization-aware training and model conversion.
"""

import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.ao.quantization as tq
import torch.nn as nn
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from torchvision.models.quantization.mobilenetv3 import (
    _mobilenet_v3_conf,
    _mobilenet_v3_model,
)

try:
    # Newer PyTorch location
    from torch.ao.nn.intrinsic.qat import freeze_bn_stats
except ImportError:  # pragma: no cover - fallback for older PyTorch versions
    from torch.nn.intrinsic.qat import freeze_bn_stats

from utils.model import save_model, train_single_epoch, validate_single_epoch

# Backends whose int8 QEngine we support. "x86" is preferred on modern x86
# CPUs; "fbgemm" is the legacy x86 path and "qnnpack" targets ARM/mobile.
_SUPPORTED_BACKENDS = ("x86", "fbgemm", "qnnpack")


def _build_classifier_head(
    in_features: int, num_classes: int, dropout_rate: float
) -> nn.Sequential:
    """Build the household-objects classifier head (Linear -> Hardswish -> ...)."""
    return nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(inplace=True),
        nn.Dropout(p=dropout_rate, inplace=True),
        nn.Linear(1024, num_classes),
    )


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
        
        # Create a quantizable MobileNetV3 Small
        inverted_residual_setting, last_channel = _mobilenet_v3_conf("mobilenet_v3_small")
        self.model = _mobilenet_v3_model(
            inverted_residual_setting=inverted_residual_setting,
            last_channel=last_channel,
            weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None,
            progress=True,
            quantize=quantize,
        )
        
        # Swap in a task-specific classifier head for the household dataset.
        head_in = self.model.classifier[0].in_features
        self.model.classifier = _build_classifier_head(
            head_in, num_classes, dropout_rate
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
        # torchvision's quantizable MobileNetV3 (self.model) already ships
        # with its own `fuse_model` implementation that knows how to fuse
        # its Conv+BN(+ReLU) blocks (and, when is_qat=True, uses the
        # QAT-specific fused+observed module types instead of the
        # inference-only fused ones).
        self.model.fuse_model(is_qat=is_qat)

        # Note: the custom classifier head uses Linear -> Hardswish, and
        # Hardswish is not one of the patterns `fuse_modules`/`fuse_modules_qat`
        # supports, so there is nothing further to fuse there.
        return self


def _prepare_for_qat(model: nn.Module, backend: str = "fbgemm") -> nn.Module:
    """Insert fake-quant observers so ``model`` can be fine-tuned in QAT.

    Steps (in order): validate/select the QEngine, switch to train mode, fuse
    Conv+BN(+ReLU) with the QAT-aware fused modules, attach the backend's
    default QAT qconfig, then swap eligible float modules for their observed
    fake-quant counterparts.

    Args:
        model: Float model exposing a ``fuse_model(is_qat=...)`` method.
        backend: One of ``"x86"`` (recommended on x86 CPUs since PyTorch 2.0 --
            oneDNN/FBGEMM dispatch, ~1.43x faster int8 than plain ``"fbgemm"``),
            ``"fbgemm"`` (legacy x86) or ``"qnnpack"`` (ARM/mobile).

    Returns:
        The same model instance, now prepared for QAT (modified in place).
    """
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(
            "Backend must be one of 'x86'/'fbgemm' (x86 CPU) or 'qnnpack' (ARM)"
        )

    torch.backends.quantized.engine = backend
    model.train()
    model.fuse_model(is_qat=True)
    model.qconfig = tq.get_default_qat_qconfig(backend)
    tq.prepare_qat(model, inplace=True)
    return model


def _convert_to_int8(model: nn.Module) -> nn.Module:
    """Materialize a real int8 model from a (fake-quant) QAT model.

    Quantized kernels only execute on CPU in eval mode, so the model is moved to
    CPU and switched to eval before ``convert`` replaces the observed modules
    with their true int8 implementations.

    Args:
        model: A QAT-prepared (and ideally fine-tuned) model.

    Returns:
        A freshly converted int8 model (the input is left untouched).
    """
    model = model.cpu()
    model.eval()
    return tq.convert(model, inplace=False)


def _rebuild_optimizer(
    optimizer: torch.optim.Optimizer, params: Any
) -> torch.optim.Optimizer:
    """Recreate ``optimizer`` against ``params`` reusing its hyperparameters.

    ``prepare_qat`` swaps modules in place and can allocate brand-new parameter
    tensors (observers, fake-quant scale/zero-point buffers, fused-module
    weights), leaving the original optimizer tracking stale references. A fresh
    optimizer of the same class -- built from a *copy* of ``defaults`` so the
    original dict is never mutated -- keeps training pointed at live tensors.
    """
    hyperparams = dict(optimizer.defaults)
    hyperparams.pop("decoupled_weight_decay", None)
    return optimizer.__class__(params, **hyperparams)


def _rebuild_onecycle(
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.OneCycleLR:
    """Clone a OneCycleLR onto ``optimizer`` preserving its full shape.

    The one-cycle envelope (``max_lr``, ``pct_start`` and the two div factors)
    is recovered from the existing schedule/optimizer instead of falling back to
    library defaults, so the post-``prepare_qat`` cycle matches the configured
    one.
    """
    group = scheduler.optimizer.param_groups[0]
    max_lr = group.get("max_lr", optimizer.defaults["lr"])
    initial_lr = group.get("initial_lr", max_lr / 25.0)
    min_lr = group.get("min_lr", initial_lr / 1e4)
    try:
        pct_start = (
            scheduler._schedule_phases[0]["end_step"]
            / (scheduler.total_steps - 1)
        )
    except (AttributeError, IndexError, KeyError, ZeroDivisionError):
        pct_start = 0.3
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=pct_start,
        div_factor=max_lr / initial_lr,
        final_div_factor=initial_lr / min_lr,
    )


def _reanchor_scheduler(
    scheduler: Optional[Any],
    optimizer: torch.optim.Optimizer,
    remaining_epochs: int,
    steps_per_epoch: int,
) -> Optional[Any]:
    """Re-point ``scheduler`` at a freshly rebuilt ``optimizer``.

    Each scheduler family needs slightly different handling: OneCycleLR is
    rebuilt for the remaining horizon, ReduceLROnPlateau/CosineAnnealingLR are
    reconstructed from their public knobs, and anything else is rebuilt
    generically from its own ``__init__`` signature (dropping ``optimizer`` and
    ``last_epoch`` to avoid the resume path that demands ``initial_lr``).
    """
    if scheduler is None:
        return None

    sched = torch.optim.lr_scheduler
    if isinstance(scheduler, sched.OneCycleLR):
        return _rebuild_onecycle(
            scheduler, optimizer, remaining_epochs, steps_per_epoch
        )
    if isinstance(scheduler, sched.ReduceLROnPlateau):
        return sched.ReduceLROnPlateau(optimizer, mode="min")
    if isinstance(scheduler, sched.CosineAnnealingLR):
        return sched.CosineAnnealingLR(
            optimizer, T_max=scheduler.T_max, eta_min=scheduler.eta_min
        )

    ctor_params = scheduler.__init__.__code__.co_varnames
    return scheduler.__class__(optimizer, **{
        key: value
        for key, value in vars(scheduler).items()
        if key in ctor_params and key not in ("self", "optimizer", "last_epoch")
    })


def _is_one_cycle(scheduler: Optional[Any]) -> bool:
    """True when ``scheduler`` must be stepped every batch (OneCycleLR)."""
    return isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR)


def _score_int8_snapshot(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    epoch: int,
    num_epochs: int,
) -> Tuple[float, float]:
    """Evaluate a *converted* int8 snapshot of ``model`` on CPU.

    A CPU deep copy is converted to true int8 so the reported loss/accuracy
    reflect the model that will actually be deployed. The temporary copies are
    dropped before returning to keep peak memory flat across epochs.
    """
    snapshot = copy.deepcopy(model).cpu()
    snapshot.eval()
    int8_model = _convert_to_int8(snapshot)
    loss, accuracy = validate_single_epoch(
        int8_model, loader, criterion, torch.device("cpu"), epoch, num_epochs
    )
    del snapshot, int8_model
    return loss, accuracy


class _TrainingHistory:
    """Thin ordered-column recorder for per-epoch training metrics."""

    _COLUMNS = (
        "epoch", "train_loss", "train_accuracy",
        "test_loss", "test_accuracy", "epoch_time", "lr",
    )

    def __init__(self) -> None:
        self._columns: Dict[str, List[Any]] = {c: [] for c in self._COLUMNS}

    def append(self, **values: Any) -> None:
        for column in self._COLUMNS:
            self._columns[column].append(values[column])

    def to_dict(self) -> Dict[str, List[Any]]:
        return self._columns


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
    cfg = training_config
    num_epochs = cfg.get("num_epochs", 100)
    criterion = cfg.get("criterion")
    optimizer = cfg.get("optimizer")
    scheduler = cfg.get("scheduler")
    patience = cfg.get("patience", 10)
    device = cfg.get(
        "device",
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    grad_clip_norm = cfg.get("grad_clip_norm", None)
    freeze_bn_epochs = cfg.get("freeze_bn_epochs", 0)
    qat_start_epoch = cfg.get("qat_start_epoch", 0)
    # Freeze observers halfway through the QAT window unless told otherwise, so
    # activation quant ranges settle before the schedule ends.
    observer_freeze_epoch = cfg.get(
        "observer_freeze_epoch",
        qat_start_epoch + max(1, (num_epochs - qat_start_epoch) // 2),
    )
    steps_per_epoch = len(train_loader)

    print(f"Training with quantization-aware training for {num_epochs} epochs")
    print(
        f"QAT start epoch: {qat_start_epoch}, "
        f"Finetune BN stats epochs: {freeze_bn_epochs}"
    )
    print(f"QAT will be activated after epoch {qat_start_epoch}")

    history = _TrainingHistory()
    best_accuracy = 0.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        qat_live = epoch >= qat_start_epoch

        # --- Transition fp32 -> QAT exactly once, re-anchoring optim/sched. ---
        if epoch == qat_start_epoch:
            print(f"Activating QAT at epoch {epoch + 1}")
            model = _prepare_for_qat(model, backend=backend)
            model.to(device)
            optimizer = _rebuild_optimizer(optimizer, model.parameters())
            scheduler = _reanchor_scheduler(
                scheduler, optimizer, num_epochs - qat_start_epoch, steps_per_epoch
            )

        # OneCycleLR advances per batch inside the epoch trainer; every other
        # scheduler is epoch-stepped after evaluation.
        batch_scheduler = scheduler if _is_one_cycle(scheduler) else None
        train_loss, train_accuracy = train_single_epoch(
            model, train_loader, criterion, optimizer, device,
            grad_clip_norm=grad_clip_norm, epoch=epoch, num_epochs=num_epochs,
            scheduler=batch_scheduler,
        )

        # Stabilize quantization once enough QAT fine-tuning has happened.
        if qat_live and epoch == observer_freeze_epoch:
            print(f"Disabling observers at epoch {epoch + 1}")
            model.apply(tq.disable_observer)
        if freeze_bn_epochs > 0 and epoch == qat_start_epoch + freeze_bn_epochs:
            print(f"Freezing BatchNorm running stats at epoch {epoch + 1}")
            model.apply(freeze_bn_stats)

        # Score the int8 model once QAT is live, else the plain fp32 model.
        if qat_live:
            test_loss, test_accuracy = _score_int8_snapshot(
                model, test_loader, criterion, epoch, num_epochs
            )
        else:
            test_loss, test_accuracy = validate_single_epoch(
                model, test_loader, criterion, device, epoch, num_epochs
            )

        # Advance epoch-stepped schedulers (OneCycleLR already stepped above).
        if scheduler is not None and not _is_one_cycle(scheduler):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        epoch_time = time.time() - epoch_start_time
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
            f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, "
            f"LR: {lr:.6f}, Time: {epoch_time:.2f}s"
        )

        # Checkpoint the best int8 model; only count/stop stalls once QAT runs
        # so early stopping can never fire before a checkpoint exists.
        if qat_live and test_accuracy > best_accuracy:
            print(f"New best quantized model! Saving... ({test_accuracy:.2f}%)")
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            save_model(model, checkpoint_path)
            stale_epochs = 0
        elif qat_live:
            stale_epochs += 1

        history.append(
            epoch=epoch + 1,
            train_loss=train_loss,
            train_accuracy=train_accuracy,
            test_loss=test_loss,
            test_accuracy=test_accuracy,
            epoch_time=epoch_time,
            lr=lr,
        )

        # Break after recording so the final epoch's metrics are kept.
        if qat_live and stale_epochs >= patience:
            print(
                f"Early stopping at epoch {epoch+1}. "
                f"No improvement for {patience} epochs."
            )
            break

    training_stats = history.to_dict()
    print(f"Training completed. Best accuracy: {best_accuracy:.2f}%")
    print(f"Best QAT model saved as '{checkpoint_path}' at epoch {best_epoch}")

    print("Converting best QAT model to fully quantized model...")
    # Restore the best checkpoint when one exists; otherwise convert whatever is
    # in memory (guards the degenerate no-improvement run from FileNotFoundError).
    if os.path.exists(checkpoint_path):
        model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
    else:
        print(
            f"WARNING: no checkpoint found at '{checkpoint_path}'; converting the "
            "current model instead (no QAT epoch improved on the initial accuracy)."
        )
    quantized_model = _convert_to_int8(model)

    return quantized_model, training_stats, best_accuracy, best_epoch