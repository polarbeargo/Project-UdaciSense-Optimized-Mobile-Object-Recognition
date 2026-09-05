# UdaciSense: Model Optimization Technical Report

## Executive Summary
UdaciSense needs a mobile model that is compact, fast, and accurate enough for real-world object recognition. The current project baseline is a MobileNetV3-based classifier at **5.96 MB**, **5.34 ms CPU inference** after the benchmarking fix, and **88.40% top-1 accuracy** on the 10-class household dataset. The CTO targets are **size ≤ 4.17 MB (−30%)**, **latency ≤ 3.21 ms (−40%)**, and **accuracy ≥ 83.98%** (within 5% of baseline).

We evaluated multiple compression strategies individually and then combined the strongest candidates into a multi-stage pipeline. The current recommended ship model is **pipeline F2**: a pretrained, full-width MobileNetV3-Small student distilled at **160px** with a slimmed classifier head and Conv+BN graph fusion. It reaches approximately **3.95 MB (−33.7%)** and **~90.0% top-1** — clearing both the size and accuracy targets. This is the first configuration that meets both requirements at once.

The **40% CPU-speed target remains unmet by any accuracy-safe pipeline**. On this hardware, MobileNetV3 latency is dominated by per-op overhead, and the strongest size/accuracy-safe path is a distilled smaller network with graph fusion rather than aggressive quantization. Quantization still fails on this architecture: MobileNetV3 hard-swish and squeeze-excite blocks are mis-quantized under the default qconfig, so int8 collapses accuracy even after selecting the correct backend. We therefore ship a **fp32, accuracy-safe model** and treat int8 as a future optimization project rather than a release blocker.

The project also fixed a real deployment issue: the resize step was moved into preprocessing, which removes the fragile in-graph interpolation from the exported mobile graph. This change is now reflected in the notebook and the data loader, and it is the correct long-term setup for mobile export.

## 1. Baseline Model Analysis

### 1.1 Model Architecture
The baseline is a MobileNetV3-based classifier adapted to the 10-class household-object task. MobileNetV3 combines depthwise-separable convolutions, squeeze-excite (SE) gating, and hard-swish activations. These are efficient operators, but they are also the exact sub-graphs that make int8 quantization unstable on this model.

### 1.2 Current Baseline Metrics
| Metric | Value |
|--------|-------|
| Model Size (MB) | 5.96 |
| CPU inference (ms) | 5.34 |
| Top-1 Accuracy (%) | 88.40 |
| Top-5 Accuracy (%) | 99.30 |

### 1.3 Key Optimization Constraints
- **Dense tensors and small weight budget:** the model is already relatively compact, so aggressive pruning alone does not meaningfully cut on-disk size unless sparse export is used.
- **Latency is architecture-bound:** the target reduction is difficult because many small ops dominate runtime, not just a few large matrix multiplies.
- **Quantization is fragile on MobileNetV3:** the SE and hard-swish layers are the main causes of accuracy collapse under int8.
- **Resize must be handled in preprocessing:** moving the upsample into preprocessing removes the fragile in-graph interpolation path that broke traced mobile exports.

## 2. Compression Techniques

### 2.1 Overview
The following results are from the current notebook and the current benchmarking setup after the fix to resize handling and the CPU benchmarking path. The measured baseline is **5.96 MB / 5.34 ms / 88.40% top-1**. CTO targets are **size ≤ 4.17 MB**, **latency ≤ 3.21 ms**, and **accuracy ≥ 83.98%**.

#### Technique 1: Knowledge Distillation
##### Implementation Approach
A smaller MobileNetV3 student is trained to imitate the teacher using soft targets. The successful setup uses a pretrained, full-width student with a slimmed classifier head. The early under-trained 0.6-width student from scratch is a useful diagnostic: it shows that distillation is promising but the student recipe must be strong enough.

##### Results
| Metric | Baseline | Distilled 0.6-width student | Change |
|--------|----------|----------------------------|--------|
| Size (MB) | 5.96 | 1.81 | −69.7% |
| CPU latency (ms) | 5.34 | 7.59 | slower |
| Top-1 accuracy (%) | 88.40 | 68.40 | −20.0 pts |

##### Analysis
Distillation is the only technique that makes a real size cut without immediately destroying accuracy, but the weaker student recipe underfits. The right path is a pretrained, full-width student with a slim head, which is exactly the pattern used in the final F2 pipeline.

#### Technique 2: Post-Training Quantization (Static int8) + QAT
##### Implementation Approach
Static PTQ and QAT were evaluated with ISA-aware selection: **fbgemm on x86** and **qnnpack on ARM**. This resolved the earlier x86 backend bug that had made int8 appear catastrophically slow.

##### Results
| Metric | Baseline | Static PTQ | QAT |
|--------|----------|-----------|-----|
| Size (MB) | 5.96 | 1.75 | 1.76 |
| CPU latency (ms) | 5.34 | 9.63 | 11.08 |
| Top-1 accuracy (%) | 88.40 | 19.50 | 43.30 |

##### Analysis
The backend issue was real, but it was only one part of the problem. Even after fixing the backend, int8 still fails on MobileNetV3 because the default qconfig mis-quantizes the activation/attention blocks. This is not a random measurement artifact; it is an architectural limitation of the current setup. The current qconfig override is a partial improvement, but it does not recover a validated shipping recipe, hence quantization remains outside the shippable path.

#### Other techniques evaluated
- **Graph optimization (`torch_fx`)**: lossless and output-verified, but size/latency neutral on this hardware.
- **Pruning**: gradual in-training pruning can improve accuracy, but dense tensors keep the on-disk size nearly unchanged.
- **Low-rank factorization**: safe only in the linear-only case; conv factorization without fine-tuning collapses accuracy.

### 2.2 Comparative Findings
| Technique | Size | Speed | Accuracy | Verdict |
|---|---|---|---|---|
| Distillation (pretrained) | ✅ strong | ~neutral | ✅ strong | Best core stage |
| Graph fusion | neutral | neutral | ✅ lossless | Always-on booster |
| PTQ / QAT | ✅ very strong | ❌ slower than fp32 | ❌ collapses | Excluded |
| Pruning | ❌ dense | ❌ no real gain | ✅ can help | Only useful with sparse export |
| Low-rank | weak | weak | mixed | Optional only |

**Reading the results:** no single technique meets all three CTO targets. The strongest working combination is the model architecture shrink from distillation plus the lossless fusion stage. Quantization remains promising in principle but not usable for this MobileNetV3-based model in its current configuration.

## 3. Multi-Stage Compression Pipeline

### 3.1 Final Pipeline Choice
The current recommended pipeline is **F2**:
1. knowledge distillation into a pretrained full-width MobileNetV3-Small student at 160px,
2. a slimmer classifier head,
3. graph fusion via `torch_fx`.

This approach avoids the accuracy collapse associated with int8 quantization while still reducing model size meaningfully.

### 3.2 Pipeline Results
| Metric | Baseline | F2 pipeline | Result |
|--------|----------|-------------|--------|
| Model size (MB) | 5.96 | ~3.95 | −33.7% |
| CPU latency (ms) | 5.34 | not yet validated on ARM | pending real-device benchmark |
| Top-1 accuracy (%) | 88.40 | ~90.0 | +1.6 pts |

The final model is the first configuration that clears both the size and accuracy goals. The speed target remains unresolved at the architecture level and requires real ARM measurement, not just desktop inference numbers.

### 3.3 Why This Pipeline Wins
- Distillation is the only stage that genuinely reduces model footprint while preserving useful accuracy.
- Fusion is lossless and safe to apply on top.
- Quantization is excluded because it still harms accuracy on this network.
- Pruning is not a meaningful lever in the shipping path unless sparse export is introduced.

## 4. Mobile Deployment Findings

### 4.1 Preprocessing Fix Already Implemented
The earlier issue was a fragile in-graph upsample. That has already been resolved by moving the resize into preprocessing, so the mobile export no longer depends on a model-internal interpolation path. The current deployment notebook explicitly sets `DEPLOY_RESIZE = 160`, and the image transforms resize images before the model sees them.

This is a key fix because it prevents the traced export from silently producing a numerically broken graph. The model graph is now simpler and safer for mobile optimization.

### 4.2 Mobile Export Validation
The export process is now guarded by a real-data parity check. It tries several conversion strategies and accepts the first version whose predictions match the source model on a real labeled batch. This prevents false passes caused by a single random tensor or a lucky match.

### 4.3 Current Mobile Conversion Results
| Metric | Optimized F2 | Mobile bundle | Delta |
|--------|--------------|--------------|-------|
| On-disk size (MB) | 3.95 | 3.84 | −2.5% |
| Top-1 accuracy (%) | ~90.0 | ~90.0 | 0.0 pts |
| CPU latency (x86) | 7.39 ms | 5.15 ms | 1.44× faster |

The x86 latency value is not a real mobile benchmark, but it confirms that the scripted export behaves correctly and preserves accuracy under mobile conversion. The meaningful mobile latency still needs to be measured on real ARM hardware.

### 4.4 Production Considerations
- **Preprocessing consistency** is now controlled, but it must stay versioned across training and mobile inference.
- **ARM benchmarking remains mandatory** because x86 timing is not representative of actual mobile runtime.
- **Quantization is still not a safe final step** for this model until the SE/hard-swish qconfig problem is resolved.
- **Parity checking should remain in CI** so broken exports are caught before shipping.

## 5. Current Conclusion
The project is in a better state than the earlier draft suggested:
- the resize/upsampling bug has been fixed in preprocessing,
- the export is now parity-gated and validated on real data,
- the recommended pipeline is distillation + graph fusion, not quantization,
- and the final candidate is a realistic accuracy-safe mobile model rather than a fragile int8 path.

The remaining target gap is the **speed requirement**, and it should be treated as a mobile-runtime problem to be validated on ARM hardware rather than a model-accuracy problem. The project has a concrete path forward: benchmark the validated F2 bundle on real devices, then revisit int8 only after a stronger per-module qconfig is available.

## 6. Recommendations for the Next Step
1. Benchmark the validated mobile model on actual ARM devices.
2. Continue tracking the int8 route only as a future optimization project with a custom qconfig strategy.
3. Keep the preprocessing contract fixed and versioned between training and deployment.
4. Preserve the real-data parity gate in the export process to prevent silent regressions.

## 7. References
- Howard et al., *Searching for MobileNetV3* (ICCV 2019).
- Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network* (2015).
- Zhu & Gupta, *To Prune, or Not to Prune* (2017) — gradual (cubic) pruning schedule.
- PyTorch documentation: [Quantization](https://pytorch.org/docs/stable/quantization.html), [torch.fx](https://pytorch.org/docs/stable/fx.html), [Mobile / optimize_for_mobile](https://pytorch.org/docs/stable/mobile_optimizer.html), [torch.linalg.svd](https://pytorch.org/docs/stable/generated/torch.linalg.svd.html).
