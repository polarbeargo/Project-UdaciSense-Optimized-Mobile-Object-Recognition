import os
import json
import time
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.mobilenetv3 import _mobilenet_v3_conf, MobileNetV3
from torch.nn import functional as F
from tqdm import tqdm
from typing import Dict, Any, Tuple, Optional, List

from utils.model import *

# Student model: a slimmer MobileNetV3-Small created by shrinking the width
# multiplier and classifier size. A narrower network has far fewer parameters
# and FLOPs than the teacher, directly reducing model size and inference time;
# knowledge distillation is what lets it recover most of the teacher's accuracy.
class MobileNetV3_Household_Small(nn.Module):
    """
    Student model based on MobileNetV3-Household.
    """
    
    def __init__(self, num_classes=10, width_mult=0.6, linear_size=256, dropout=0.2, input_size=224, pretrained=False):
        super().__init__()

        # Persist hyperparameters so checkpoints can be reloaded with the exact
        # same architecture (see train_with_distillation's load_model call).
        self.width_mult = width_mult
        self.linear_size = linear_size
        self.dropout = dropout
        self.num_classes = num_classes
        self.pretrained = pretrained
        # Spatial resolution the backbone actually runs at. The forward pass
        # upsamples the incoming image to this size, and this F.interpolate is the
        # dominant CPU cost -- lowering input_size (e.g. 224 -> 112) cuts inference
        # latency roughly quadratically, which is the main lever for the CPU-speed
        # target. Persisted so checkpoints reload with the identical geometry.
        self.input_size = input_size

        # Build a reduced-width MobileNetV3-Small backbone.
        inverted_residual_setting, last_channel = _mobilenet_v3_conf(
            "mobilenet_v3_small", width_mult=width_mult
        )
        self.model = MobileNetV3(
            inverted_residual_setting,
            last_channel,
            num_classes=num_classes,
            dropout=dropout,
        )

        # Optional transfer learning: at width_mult=1.0 the backbone geometry is
        # identical to torchvision's mobilenet_v3_small, so we can warm-start the
        # feature extractor from ImageNet weights instead of training from scratch.
        # A from-scratch student plateaus well below the accuracy target; ImageNet
        # initialization gives the distillation a much stronger starting point.
        # Only the feature layers are copied (matching shapes); the custom
        # classifier head below is always trained fresh.
        if pretrained:
            if abs(width_mult - 1.0) > 1e-6:
                raise ValueError(
                    "pretrained=True requires width_mult=1.0 (torchvision only "
                    f"ships ImageNet weights for the full-width backbone); got width_mult={width_mult}."
                )
            pretrained_sd = models.mobilenet_v3_small(weights="DEFAULT").state_dict()
            own_sd = self.model.state_dict()
            transferable = {
                k: v for k, v in pretrained_sd.items()
                if k in own_sd and own_sd[k].shape == v.shape
            }
            self.model.load_state_dict(transferable, strict=False)

        # Replace the classifier head with a configurable hidden size.
        in_features = self.model.classifier[0].in_features
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, linear_size),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(linear_size, num_classes),
        )
    
    def forward(self, x):
        # Ensure input is correctly sized. Skip the interpolate entirely when the
        # incoming spatial size already matches, avoiding a needless (and costly)
        # resample on the CPU inference path.
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = F.interpolate(
                x, size=(self.input_size, self.input_size),
                mode='bilinear', align_corners=False,
            )
        return self.model(x)
    
    
# Compute the knowledge distillation loss.
# `temperature` softens both probability distributions so the student learns
# from the teacher's inter-class similarities ("dark knowledge"); `alpha`
# balances the soft-target KL term against the hard-label cross-entropy term.
def _knowledge_distillation_loss(student_logits, teacher_logits, targets, temperature=2.0, alpha=0.5):
    """
    Compute the knowledge distillation loss.
    
    Args:
        student_logits: Output logits from the student model
        teacher_logits: Output logits from the teacher model
        targets: Ground truth labels
        temperature: Temperature for softening the teacher's probability distribution
        alpha: Weight balancing cross entropy and distillation loss
        
    Returns:
        Final loss combining distillation and standard cross entropy
    """
    # Soft-target loss: KL divergence between the temperature-softened teacher
    # and student distributions. Multiplying by T^2 keeps gradient magnitudes
    # comparable to the hard-label term (as in Hinton et al., 2015).
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    distillation_loss = F.kl_div(
        soft_student, soft_teacher, reduction="batchmean"
    ) * (temperature ** 2)

    # Hard-target loss: standard cross entropy against the ground-truth labels.
    student_loss = F.cross_entropy(student_logits, targets)

    return alpha * distillation_loss + (1.0 - alpha) * student_loss


@torch.no_grad()
def recalibrate_bn(
    model: nn.Module,
    data_loader,
    device: torch.device,
    num_batches: int = 50,
) -> nn.Module:
    """Re-estimate BatchNorm running statistics for a trained model.

    A student built from scratch accumulates BatchNorm ``running_mean`` /
    ``running_var`` using torchvision MobileNetV3's default momentum (0.01). Those
    running stats can stay poorly matched to the final trained weights, which makes
    the network behave in ``.train()`` (per-batch stats) but collapse toward a
    uniform output in ``.eval()`` (stale running stats) -- the exact train-good /
    eval-chance signature we observed. Resetting the running stats and recomputing
    them over a few hundred training samples with the *current* weights fixes the
    train/eval mismatch without taking any additional gradient steps.

    Args:
        model: Model whose BatchNorm running stats should be recomputed.
        data_loader: Data loader providing representative (training) inputs.
        device: Device to run the calibration forward passes on.
        num_batches: Number of batches to average running statistics over.

    Returns:
        The same model with refreshed BatchNorm running statistics.
    """
    bn_modules = [
        m for m in model.modules()
        if isinstance(m, nn.modules.batchnorm._BatchNorm)
    ]
    if not bn_modules:
        return model

    was_training = model.training
    # Eval mode disables dropout so calibration activations are noise-free; each
    # BatchNorm layer is individually switched to train mode so it still updates
    # its running stats, and momentum=None gives an unbiased cumulative average.
    model.eval()
    saved_momentum = {}
    for m in bn_modules:
        m.reset_running_stats()
        saved_momentum[m] = m.momentum
        m.momentum = None
        m.train()

    for i, (inputs, _) in enumerate(data_loader):
        if i >= num_batches:
            break
        model(inputs.to(device))

    # Restore original momentum and the model's previous train/eval mode.
    for m in bn_modules:
        m.momentum = saved_momentum[m]
    model.train(was_training)
    return model


def _distill_single_epoch(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temperature: float = 2.0,
    alpha: float = 0.5,
    grad_clip_norm: Optional[float] = None,
    epoch: Optional[int] = None,
    num_epochs: Optional[int] = None
) -> Tuple[float, float]:
    """
    Train a student model for a single epoch using knowledge distillation.
    
    Args:
        student_model: Smaller model to be trained (student)
        teacher_model: Larger pre-trained model (teacher)
        train_loader: Training data loader
        criterion: Standard loss function (for hard targets)
        optimizer: PyTorch optimizer for student model
        device: Device to train on
        temperature: Temperature for softening probability distributions
        alpha: Weight for balancing hard and soft targets
        grad_clip_norm: Maximum norm for gradient clipping (optional)
        epoch: Current epoch (optional, for progress display)
        num_epochs: Total number of epochs (optional, for progress display)
        
    Returns:
        Tuple of (train_loss, train_accuracy)
    """
    # Set models to appropriate modes
    student_model.train()
    teacher_model.eval()  # Teacher is always in eval mode
    
    # Initialize tracking variables
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    # Progress bar description
    if epoch is not None and num_epochs is not None:
        desc = f"Epoch {epoch+1}/{num_epochs} [Distill]"
    else:
        desc = "Distillation"
        
    pbar = tqdm(train_loader, desc=desc)
    
    # Iterate over batches
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass: student is trained, teacher only provides targets.
        optimizer.zero_grad(set_to_none=True)
        student_outputs = student_model(inputs)
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
        
        # Compute distillation loss
        loss = _knowledge_distillation_loss(
            student_outputs, 
            teacher_outputs, 
            targets, 
            temperature=temperature, 
            alpha=alpha
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping if configured
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=grad_clip_norm)
            
        # Update weights
        optimizer.step()
        
        # Update statistics
        train_loss += loss.item() * inputs.size(0)
        
        # Update accuracy metrics (based on hard predictions)
        _, predicted = student_outputs.max(1)
        batch_total = targets.size(0)
        batch_correct = predicted.eq(targets).sum().item()
            
        train_correct += batch_correct
        train_total += batch_total
        
        # Update progress bar with current batch stats
        pbar.set_postfix({
            "loss": loss.item(),
            "batch_acc": 100. * batch_correct / batch_total,
            "running_acc": 100. * train_correct / train_total,
            "lr": optimizer.param_groups[0]['lr']
        })
    
    # Calculate final metrics for training - average over all samples
    train_loss = train_loss / train_total
    train_accuracy = 100. * train_correct / train_total
    
    return train_loss, train_accuracy

def train_with_distillation(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_loader,
    test_loader,
    training_config: Dict[str, Any],
    checkpoint_path: str = 'checkpoints/distilled_model.pth'
) -> Tuple[nn.Module, Dict[str, List], float, int]:
    """
    Train a student model using knowledge distillation from a teacher model.
    
    Args:
        student_model: Smaller model to be trained (student)
        teacher_model: Larger pre-trained model (teacher)
        train_loader: Training data loader
        test_loader: Test data loader
        training_config: Dictionary containing training configuration
        checkpoint_path: Path to save the best student model
        
    Returns:
        Tuple of (student_model, training_stats, best_accuracy, best_epoch)
    """
    # Extract training configuration
    num_epochs = training_config.get('num_epochs')
    standard_criterion = training_config.get('criterion')
    optimizer = training_config.get('optimizer')
    scheduler = training_config.get('scheduler')
    patience = training_config.get('patience')
    device = training_config.get('device', 
                               torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    grad_clip_norm = training_config.get('grad_clip_norm', None)
    
    # Distillation specific parameters
    temperature = training_config.get('temperature')
    alpha = training_config.get('alpha')
    
    # Set teacher model to evaluation mode
    teacher_model.eval()
    
    # Display training configuration
    print(f"Student parameters: {count_parameters(student_model):,}")
    print(f"Teacher parameters: {count_parameters(teacher_model):,}")
    print(f"Training with knowledge distillation for {num_epochs} epochs")
    print(f"Temperature: {temperature}, Alpha: {alpha}")
    
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
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Train for one epoch with distillation
        train_loss, train_accuracy = _distill_single_epoch(
            student_model=student_model,
            teacher_model=teacher_model,
            train_loader=train_loader,
            criterion=standard_criterion,
            optimizer=optimizer,
            device=device,
            temperature=temperature,
            alpha=alpha,
            grad_clip_norm=grad_clip_norm,
            epoch=epoch,
            num_epochs=num_epochs,
        )
        
        # Recalibrate BatchNorm running statistics with the freshly-updated
        # weights before evaluating. A from-scratch student otherwise carries
        # stale running stats (momentum=0.01) that make eval-mode output collapse
        # toward uniform, so best-checkpoint selection below would be meaningless.
        recalibrate_bn(student_model, train_loader, device)

        # Evaluate student on test set (standard evaluation, no distillation)
        test_loss, test_accuracy = validate_single_epoch(
            student_model, test_loader, standard_criterion, device, epoch, num_epochs
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
        if test_accuracy > best_accuracy:
            print(f"New best student model! Saving... ({test_accuracy:.2f}%)")
            best_accuracy = test_accuracy
            best_epoch = epoch + 1
            
            save_model(student_model, checkpoint_path)
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
    
    # Load the best student model
    student_model = load_model(checkpoint_path, device, model_class=MobileNetV3_Household_Small, width_mult=student_model.width_mult, linear_size=student_model.linear_size, dropout=student_model.dropout, input_size=student_model.input_size)

    # Safety net: ensure the returned model's BatchNorm running stats match its
    # final weights so downstream eval / graph fusion / quantization see the
    # correct eval-mode behavior rather than the collapsed uniform output.
    recalibrate_bn(student_model, train_loader, device)
    student_model.eval()

    print(f"Distillation completed. Best student accuracy: {best_accuracy:.2f}%")
    print(f"Best student model saved as '{checkpoint_path}' at epoch {best_epoch}")
    
    return student_model, training_stats, best_accuracy, best_epoch