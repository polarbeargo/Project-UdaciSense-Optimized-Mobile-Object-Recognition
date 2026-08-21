# UdaciSense: Model Optimization Technical Report

## Executive Summary
UdaciSense's on-device object recognition feature needs a model that is small enough to ship in a mobile app, fast enough to feel instant, and accurate enough to be useful. The starting point was a MobileNetV3-based classifier at **5.96 MB**, **~134 ms** CPU inference, and **87.8%** top-1 accuracy on the 10-class household dataset. The CTO set three targets: **≥30% smaller**, **≥40% faster on CPU**, and **accuracy within 5%** of baseline.

We evaluated six compression techniques individually (quantization, QAT, post-training and gradual pruning, knowledge distillation, graph optimization, and low-rank factorization), then combined the winners into a multi-stage pipeline. The shipped model, **pipeline F2**, is a pretrained, full-width MobileNetV3-Small student distilled at 160px with a slimmed classifier head and Conv+BN graph fusion. It reaches **3.94 MB (−33.7%)** and **88.2% top-1 (+0.4 pts)** — clearing the **size and accuracy** targets simultaneously, the only configuration that does so.

The **40% CPU-speedup target was not met by any technique** without destroying accuracy: it is architecture-bound (MobileNetV3 latency is dominated by per-op overhead), and int8 quantization — the classic speed lever — collapses this architecture's accuracy because its hard-swish/squeeze-excite blocks are mis-quantized by the default configuration. We therefore deliberately shipped an accuracy-safe fp32 model rather than a fast-but-broken int8 one.

**Business impact:** a one-third smaller download and a numerically verified mobile bundle mean faster installs, lower bandwidth/storage cost, and no regression in recognition quality for users — while the remaining speed gap is documented with a concrete on-device roadmap (ARM benchmarking, preprocessing-side resize, and a per-module int8 qconfig).

## 1. Baseline Model Analysis

### 1.1 Model Architecture
The baseline is a **MobileNetV3-based classifier** (`MobileNetV3_Household`) adapted to the 10-class household-object task at CIFAR-scale (32×32) inputs. MobileNetV3 is an inverted-residual (MBConv) network whose blocks combine depthwise-separable convolutions, **squeeze-excite (SE)** channel attention, and **hard-swish** activations, topped by a small fully-connected classifier head. These design choices make it efficient in FLOPs, but two of them — SE gating and hard-swish — are exactly the operators that later prove fragile under int8 quantization. At **5.96 MB** the weights are already compact, so there is limited "fat" to trim without changing the architecture itself.

### 1.2 Performance Metrics
| Metric | Value |
|--------|-------|
| Model Size (MB) | 5.96 |
| Inference Time - CPU (ms) | 134 |
| Accuracy (%) | 87.8 |
| Top-5 Accuracy (%) | 99.3 |

### 1.3 Optimization Challenges
Several properties of this model shape what is achievable:
- **Already-small weights, dense tensors.** At 5.96 MB there is little redundancy to prune; and PyTorch stores pruned weights as *dense zeros*, so unstructured pruning yields ~0% on-disk size reduction unless the model is exported sparse/structured.
- **Per-op-bound latency.** MobileNetV3's runtime is dominated by many small operators rather than a few large matmuls, so operator fusion helps only modestly and the 40% CPU-speed target is hard to reach without a genuinely smaller architecture.
- **Quantization-hostile blocks.** The SE and hard-swish sub-graphs are mis-handled by the default int8 qconfig, so quantization — normally the go-to for size *and* speed — collapses accuracy on this network.
- **A resolution mismatch.** The strongest students are pretrained at higher resolution, forcing an in-graph upsample (32→160) that later becomes the fragile point during mobile conversion.

## 2. Compression Techniques 

### 2.1 Overview

All numbers below are **measured** via `compare_experiments(...)` on the deployment CPU (x86), after the int8 backend fix. Baseline reference: **5.96 MB / 134.03 ms / 87.80% top-1**. Targets: size ≤ 4.17 MB, latency ≤ 80.4 ms, accuracy ≥ 83.41%.

#### Technique 1: Knowledge Distillation
##### Implementation Approach
A smaller MobileNetV3-Small student is trained to mimic the baseline teacher using soft targets (`temperature=4.0`, `alpha=0.7`, T²-scaled KD loss + 30% hard-label CE), AdamW with cosine schedule for 30 epochs. The slimmer student is what actually reduces size/latency; distillation is how accuracy is (partly) recovered. The initial student used `width_mult=0.6`, `linear_size=256`.

##### Results
| Metric | Baseline | After Distillation (0.6-width) | Change (%) |
|--------|----------|-------------------|------------|
| Model Size (MB) | 5.96 | 1.81 | −69.7% |
| Inference Time - CPU (ms) | 134.03 | 127.95 | −4.5% |
| Accuracy (%) | 87.80 | 67.10 | −20.7 pts |
| Top-1 vs target | ≥ 83.41% | 67.10% | Fails accuracy floor |

##### Analysis
Distillation is the **only technique that cuts size dramatically while keeping latency flat** — the ideal profile — but the 0.6-width student trained from scratch is under-fit at 30 epochs (67.1%). The method is right; the *recipe* is the limiter. This motivated the pipeline's key move: distill a **pretrained, full-width** student instead (see §3), which recovers accuracy while still shrinking the model.

#### Technique 2: Post-Training Quantization (Static int8) + QAT
##### Implementation Approach
Static PTQ quantizes weights and activations to int8 after a short calibration pass; QAT simulates quantization during an 8-epoch fine-tune. Both select the CPU backend by ISA — **`fbgemm` on x86, `qnnpack` on ARM** — after we diagnosed that the earlier "prefer qnnpack" heuristic was picking slow reference kernels on x86.

##### Results
| Metric | Baseline | Static PTQ | QAT |
|--------|----------|-----------|-----|
| Model Size (MB) | 5.96 | 1.75 (−70.6%) | 1.76 (−70.5%) |
| Inference Time - CPU (ms) | 134.03 | 115.03 (1.2×) | 99.02 (1.4×) |
| Accuracy (%) | 87.80 | 16.50 | 39.10 |

##### Analysis
Int8 delivers the best **size and speed** of any technique (up to 1.4× faster, ~70% smaller), and the earlier ~36× *slowdown* was purely a **backend-selection bug**, now fixed. What remains is an **architectural accuracy collapse**: MobileNetV3's hard-swish/SE blocks are mis-quantized by the default qconfig, dropping top-1 to 16.5% (PTQ) / 39.1% (QAT). Quantization is therefore unusable here without a per-module qconfig, and it is deliberately excluded from the shipped pipeline.

#### Additional techniques evaluated
- **Graph optimization (torch_fx):** fuses Conv+BN and strips dropout; **output-verified lossless** (87.8% preserved), latency-neutral on this hardware but composes safely on top of anything — kept as an always-on stage.
- **Post-training pruning (0.5) & gradual in-training pruning (0.6):** gradual pruning actually **improves accuracy** (89.3%, +1.5 pts from co-adaptation) but neither shrinks the dense `.pth` (0.0–0.1% size change); useful only with sparse/structured export.
- **Low-rank factorization (SVD):** Linear-only is safe but low-yield (87.7% at −1.4% size); factorizing the conv1×1 backbone without fine-tuning collapses accuracy to ~10%.

### 2.2 Comparative Analysis
| Technique | Size lever | Speed lever | Accuracy-safe? | Verdict |
|---|---|---|---|---|
| Distillation (pretrained) | ✅ strong | ✅ (smaller net) | ✅ with right recipe | **Core stage** |
| Graph optimization (fx) | – | ~neutral | ✅ lossless | **Always-on booster** |
| Quantization / QAT | ✅ strong | ✅ strong | ❌ collapses on MobileNetV3 | Excluded until qconfig fixed |
| Gradual pruning | ❌ (dense) | ❌ | ✅ (+accuracy) | Only with sparse export |
| Low-rank (Linear) | weak | weak | ✅ | Optional add-on |
| Low-rank (conv) | moderate | ❌ | ❌ | Needs fine-tune |

**Takeaway:** the only pair that moves **size without wrecking accuracy** is **distillation + graph fusion**. Quantization is the highest-upside lever but is blocked by an accuracy (not speed) issue. This directly shapes the pipeline design.

## 3. Multi-Stage Compression Pipeline

### 3.1 Pipeline Design
Guided by §2, the pipeline avoids quantization and centers on a **distilled smaller architecture** followed by **lossless graph fusion**. We iterated through several pipeline variants (A–F) and converged on **F2**:
1. **Knowledge distillation** into a **pretrained, full-width (`width_mult=1.0`) MobileNetV3-Small** student at **160px**, with a **slimmed classifier head (`linear_size=128`)** — the pretrained backbone recovers accuracy that the from-scratch 0.6-width student could not, and the slim head is what pushes size below the 30% bar.
2. **Graph optimization (torch_fx)** — Conv+BN fusion, applied losslessly on top.

### 3.2 Implementation
Each stage is an isolated, checkpointed step so intermediate models and metrics are saved and comparable. The distillation stage reuses the Notebook 02 recipe (`T=4.0`, `alpha=0.7`) with the pretrained student; the fusion stage reuses the verified `optimize_model(..., "torch_fx")` path. The final model and its metrics are persisted under `models/pipeline/pipeline_f2_pretrained_slimhead_distill_fx/` and the matching `results/` folder, which Notebook 04 loads directly. Earlier variants illustrate the design search: F (full head) reached only 4.24 MB / 28.9% (missed size); F2 slimmed the head to `linear_size=128` → 3.94 MB / 33.7% (passed).

### 3.3 Results

Final optimized model: pipeline **F2** (`pipeline_f2_pretrained_slimhead_distill_fx`) — an ImageNet-pretrained, full-width MobileNetV3-Small student, distilled at 160px with a slimmed classifier head (`linear_size=128`) and Conv+BN graph fusion (`torch_fx`).

| Metric | Baseline | Final Optimized Model (F2) | Change (%) | Requirement Met? |
|--------|----------|------------------------|------------|----------|
| Model Size (MB) | 5.96 | 3.94 | −33.7% | ✅ Yes (≥30% reduction) |
| Inference Time CPU (ms) | 134 | 120.1 | −10.4% | ❌ No (target ≥40% reduction) |
| Accuracy (%) | 87.8 | 88.2 | +0.4 pts | ✅ Yes (within 5%) |
| Top-5 Accuracy (%) | 99.3 | 99.3 | +0.0 pts | - |

F2 is the only pipeline that clears **both** the size and accuracy targets. The CPU-speed target is architecture-bound (MobileNetV3 latency is dominated by per-op overhead) and met by no pipeline; int8 quantization was rejected because it collapses MobileNetV3 accuracy via mis-quantized hard-swish/SE blocks.

### 3.4 Analysis
The pipeline succeeds because its two stages have **complementary, non-conflicting** effects. Distillation into a pretrained full-width student does the heavy lifting: it cuts size by ~34% *and* actually nudges accuracy up (+0.4 pts) because the pretrained backbone and slim head are better matched to the 10-class task than the oversized baseline head. Graph fusion then composes losslessly, guaranteeing no accuracy risk. The main trade-off encountered was **size vs. accuracy in the head width**: variant F kept the full head and missed the size bar (28.9%), while F2's `linear_size=128` head cleared it (33.7%) with no accuracy cost. The unresolved trade-off is **speed**: because we refused to trade accuracy for int8 speed, the CPU-latency target remains unmet — a deliberate, documented choice rather than an oversight.

## 4. Mobile Deployment

### 4.1 Export Process
The F2 model is exported to a mobile-ready TorchScript bundle via a **parity-gated** conversion that tries progressively safer export strategies and keeps the first whose predictions match the source model on a real test batch. The order is: `trace + optimize_for_mobile` → `script + optimize_for_mobile` → `trace + freeze` → `script + freeze`.

The naive `trace + optimize_for_mobile` path was **automatically rejected** (only 10.9% top-1 agreement — a numerical break), and **`script + optimize_for_mobile` was auto-selected** at 100% agreement. Root cause: `torch.jit.trace` + `optimize_for_mobile` mis-lowered the model's internal `F.interpolate(mode='bilinear')` upsample (32px → 160px), a known weak spot for tracing + mobile operator folding. Scripting captures the interpolate symbolically, so it keeps XNNPACK speed *and* preserves accuracy.

### 4.2 Mobile-Specific Considerations
- **CPU-only, quantization-free:** F2 runs in fp32 on the mobile CPU with no accelerator dependency, deliberately avoiding int8 (which collapses MobileNetV3 accuracy).
- **Internal upsampling is the fragile point:** the 32→160 in-graph interpolate is what broke under tracing; scripting works around it, but folding the resize into preprocessing remains the most robust long-term fix.
- **Thread & thermal sensitivity:** on real devices, latency depends on thread count and SoC throttling far more than input resolution.
- **Validation gate:** `compare_model_outputs` now evaluates both models over the real labeled test set and **fails on a >1 pt top-1 drop**, replacing a single-random-tensor check that had given a false PASS on the broken export.

### 4.3 Performance Verification
Measured on this x86 CPU workspace (latency is an on-desktop artifact — see note below):

| Metric | Source F2 (`.pth`) | Mobile bundle (`.pt`) | Δ |
|--------|--------------------|-----------------------|---|
| On-disk size (MB) | 3.94 | 3.84 | −0.10 (−2.5%) |
| Top-1 accuracy (%) | 88.2 | 88.2 | +0.00 pts |
| Top-5 accuracy (%) | 99.3 | 99.3 | +0.00 pts |
| CPU latency (ms, x86 artifact) | 120.1 | 4876 | 42× slower |

**Accuracy is fully preserved** (max logit abs diff 1.61e-04, `allclose` on 8/8 test batches, 100% prediction agreement) — the consistency check **PASSED**. The 42× latency is a measurement artifact: `optimize_for_mobile` emits XNNPACK prepacked ops tuned for ARM, which fall back to slow reference kernels on x86. Since slowness cannot alter outputs and accuracy is identical, this is a kernel-dispatch mismatch, not a correctness issue; valid latency must be measured on real ARM hardware.

### 4.4 Note: int8 accuracy diagnosis (why the mobile model stays fp32)
Quantization would be the natural way to also hit the speed target on-device, so it is worth stating precisely why F2 ships in fp32. Measured on the deployment CPU (after the backend fix), int8 delivers the size/speed but **collapses accuracy**:

| Config | Size (MB) | CPU speedup | Top-1 (%) |
|--------|-----------|-------------|-----------|
| F2 (fp32, shipped) | 3.94 | 1.0× | **88.2** |
| Static PTQ (int8) | 1.75 | 1.2× | 16.5 |
| QAT (int8) | 1.76 | 1.4× | 39.1 |

The collapse is **architectural, not a backend or calibration issue**. MobileNetV3's **squeeze-excite (SE) gating** and **hard-swish** activations are mapped to low-fidelity/unsupported patterns by PyTorch's *default* qconfig: SE multiplies two tensors whose int8 scales are mismatched, and hard-swish's piecewise curve is poorly represented at 8-bit, so error compounds across ~50 blocks. Note the earlier ~36× *slowdown* was a **separate, already-fixed** bug (qnnpack reference kernels on x86); fixing latency did not touch this accuracy problem. **Unblocking int8 requires a per-module qconfig** (keep SE/hard-swish in fp16/fp32, quantize only the linear/conv trunk) or a redesigned QAT recipe — tracked as future work, not a blocker for the fp32 deployment.

### 4.5 Cross-device benchmarking plan
The x86 latency above is not representative, so on-device numbers must be gathered on a **device-tier matrix** with a fixed, fair protocol:

| Tier | Example SoC class | Why included |
|------|-------------------|--------------|
| Low-end | entry Cortex-A53/A55 | Worst-case latency; most likely to miss the speed target |
| Mid-range | A76/A77-class big.LITTLE | Represents the median user device |
| High-end | latest flagship | Best case + headroom for future models |

For each device, collect **p50/p95/p99 single-image latency** (not just the mean), **cold-vs-warm start**, sustained **FPS**, **peak memory (RSS)**, **energy per 1k inferences**, installed **`.ptl` size**, and **on-device top-1 parity** against the desktop model on an identical held-out shard. Run **baseline vs. F2 back-to-back on the same cooled device**, with pinned thread count (`torch.set_num_threads`), fixed preprocessing/seeds, and 500+ runs after warm-up. Report distributions, and **gate release on on-device accuracy parity** so a broken conversion (like the trace path caught above) can never ship. Tools: PyTorch Mobile / ExecuTorch with Perfetto/`simpleperf` (Android) and Instruments (iOS), cross-checked against ONNX Runtime Mobile.

## 5. Conclusion and Recommendations

### 5.1 Summary of Achievements
- Shipped **pipeline F2**: **3.94 MB (−33.7%)**, **88.2% top-1 (+0.4 pts)** — meeting the **size** and **accuracy** targets, the only configuration to clear both.
- Produced a **numerically verified mobile bundle** (`script + optimize_for_mobile`) with **0.00 pt** accuracy loss vs. the source model (max logit diff 1.6e-04, allclose on all test batches).
- Diagnosed and fixed two real bugs surfaced by the data: the **int8 backend slowdown** (qnnpack-on-x86) and a **silent mobile-conversion break** (traced bilinear interpolate), the latter caught by a new real-data parity gate.

### 5.2 Key Insights
- **Measure, don't assume.** The two most important findings — the quantization slowdown and the mobile-conversion collapse — were invisible until validated on real data; a single-random-tensor smoke test gave a false PASS.
- **Separate speed bugs from accuracy bugs.** For int8, fixing the backend solved latency but left an independent, architectural accuracy problem — conflating them would have led to the wrong conclusion.
- **Architecture beats post-hoc tricks here.** A pretrained, right-sized student (distillation) achieved what pruning/quantization/low-rank could not without breaking accuracy.
- **Conversion is part of the model.** `torch.jit.script` vs `trace` materially changed correctness; export must be validated, not assumed.

### 5.3 Recommendations for Future Work
1. **Benchmark F2's script-exported bundle on real ARM** (Android/iOS) to obtain valid latency and confirm on-device accuracy parity.
2. **Move the 32→160 upsample into preprocessing** and re-export, removing the fragile in-graph interpolate so even the traced/fastest path is safe.
3. **Add a per-module int8 qconfig** (higher precision for SE/hard-swish) or a stronger QAT recipe to unlock quantization's ~1.4× speed + ~70% size on-device — the most promising route to finally hitting the 40% speed target.
4. **Export pruning sparse/structured** so its accuracy-neutral (even +) profile translates into real size/speed.
5. **Enforce the parity gate in CI** so no numerically broken bundle can ship.

### 5.4 Business Impact
A **one-third smaller** model means faster app installs/updates, lower CDN/bandwidth cost, and less device storage — all of which reduce user drop-off at download time. Because accuracy is **fully preserved end-to-end** (including through mobile conversion), recognition quality does not regress, protecting the core user experience. The remaining CPU-speed gap is bounded and comes with a concrete, low-risk roadmap (ARM benchmarking + int8 qconfig) that can deliver the speed target without a second accuracy regression — giving product and engineering a clear, de-risked next step.

## 6. References
- Howard et al., *Searching for MobileNetV3* (ICCV 2019).
- Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network* (2015).
- Zhu & Gupta, *To Prune, or Not to Prune* (2017) — gradual (cubic) pruning schedule.
- PyTorch documentation: [Quantization](https://pytorch.org/docs/stable/quantization.html), [torch.fx](https://pytorch.org/docs/stable/fx.html), [Mobile / optimize_for_mobile](https://pytorch.org/docs/stable/mobile_optimizer.html), [torch.linalg.svd](https://pytorch.org/docs/stable/generated/torch.linalg.svd.html).