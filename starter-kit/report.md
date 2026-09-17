# UdaciSense: Model Optimization Technical Report

## Executive Summary
UdaciSense needs a mobile model that is compact, fast, and accurate enough for real-world object recognition. The project baseline is a MobileNetV3-based classifier at **5.96 MB**, **5.34 ms CPU inference** after the benchmarking fix, and **88.40% top-1 accuracy** on the 10-class household dataset. The CTO targets are **size ≤ 4.17 MB (−30%)**, **CPU speedup ≥ 1.67× (≤ 3.21 ms, −40%)**, and **accuracy ≥ 83.98%** (within 5% of baseline).

We evaluated multiple compression strategies individually and then combined the strongest candidates into a multi-stage pipeline. The recommended ship model is **Pipeline G**: the baseline MobileNetV3-Small backbone put through **40% global unstructured pruning → dynamic int8 PTQ → TorchScript graph optimization** (`torch.jit.trace` → `freeze` → `optimize_for_inference`). It reaches **0.59 MB (−90.2%)**, **2.0× CPU speedup**, and **85.9% top-1 (−2.5 pts)** — the single configuration that clears **all three** CTO targets at once.

The decisive lever was **TorchScript graph optimization**. Every earlier pipeline (A–F2) satisfied at most two of three targets: the pretrained-student distillation pipelines (F, F2) cleared accuracy and size but topped out around **1.2–1.6× CPU speedup**, just under the 1.67× line, because eager-mode `torch_fx` fusion does not restructure the graph. Swapping the final stage to TorchScript — which traces to a static graph, inlines/folds constants, and fuses Conv+BN into a frozen inference-only artifact — supplied the missing size *and* speed headroom.

Quantization is now part of the shipping path, but only the **right kind**: **dynamic** int8 PTQ on the `nn.Linear` layers is lossless here, whereas **static** PTQ still collapses MobileNetV3's hard-swish / squeeze-excite ranges (~20% accuracy even on the correct backend) and is excluded. The project also fixed a real deployment issue by moving the resize step into **preprocessing**, which removes the fragile in-graph `F.interpolate` from the exported mobile graph — so Pipeline G traces cleanly.

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
- **Dense tensors and small weight budget:** the model is already relatively compact, so unstructured pruning alone does not cut on-disk size — it only pays off once a **graph runtime (TorchScript) folds the zeros in** and serializes a smaller artifact.
- **Latency is per-op-bound, not FLOP-bound:** many small ops dominate runtime, so lowering input resolution barely helped (128 vs 112 landed near the same speed). The lever that worked was **graph restructuring via TorchScript**, not fewer pixels.
- **Static quantization is fragile on MobileNetV3:** the SE and hard-swish layers cause accuracy collapse under *static* int8; **dynamic** int8 on the Linear layers is lossless and is what the shipping pipeline uses.
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
| CPU speedup | 1.0× | 1.2× | fastest single technique |
| Top-1 accuracy (%) | 88.40 | 68.20 | −20.2 pts |

##### Analysis
Distillation gives the largest single-technique size cut and the only >1.1× speedup, but the from-scratch 0.6-width student underfits badly. A pretrained, full-width student with a slim head (Pipelines F/F2) recovers accuracy to ~90%, but as a stand-alone path it never clears the speed target — which is why the shipped pipeline (G) instead prunes and TorchScript-optimizes the baseline backbone directly.

#### Technique 2: Quantization — Dynamic PTQ (lossless), Static PTQ / QAT (fragile)
##### Implementation Approach
Quantization was evaluated with ISA-aware backend selection: the unified **x86** backend on x86 and **qnnpack** on ARM. Notebook 03's Step 4.0 diagnosis proved the earlier "int8 is ~36× slower" result was purely a **backend mismatch** — the old `qnnpack`-if-listed heuristic selected slow reference kernels on x86. With the correct engine, latency is normal, which isolates accuracy as a separate, architecture-driven question.

##### Results
| Variant | Size (MB, Δ) | CPU speedup | Top-1 (%, Δ) | Verdict |
|---|---|---|---|---|
| **Dynamic PTQ** (Linear int8) | 4.24 (−28.8%) | ~1.0× | 88.40 (+0.0) | **Lossless** — used in Pipeline G |
| Static PTQ (x86, correct backend) | — | 1.40× | ~20.8 (−67.6) | Collapses; excluded |
| QAT (100-ep OneCycleLR) | 1.76 (−70.5%) | ~0.9× | 84.30 (−4.1) | Within accuracy floor, fails speed |

##### Analysis
The split is clean: **dynamic int8 on the `nn.Linear` layers is lossless** (88.40%) and becomes a core stage of the winning pipeline, while **static PTQ** cannot recover MobileNetV3's hard-swish / SE activation ranges even with the correct backend and a safe qconfig, so it stays excluded. QAT lands just inside the accuracy floor (84.30%) but never clears the 1.67× speed bar on its own. The takeaway that shaped Pipeline G: keep **dynamic** int8, drop **static** int8, and get the real size/speed win from graph optimization instead.

#### Other techniques evaluated
- **Graph optimization — `torch_fx` vs TorchScript**: `torch_fx` fusion is lossless and output-verified but size/latency-neutral here, leaving the FX pipelines at ~1.2–1.6× speed. **TorchScript** (`trace` → `freeze` → `optimize_for_inference`) is the decisive lever — it folds the pruned zeros and fuses Conv+BN into a frozen artifact, delivering the size *and* speed that push Pipeline G past all three targets.
- **Pruning**: gradual in-training pruning can slightly improve accuracy, but dense tensors keep on-disk size unchanged **until a graph runtime folds the zeros in** — which is exactly what TorchScript does in Pipeline G.
- **Low-rank factorization**: safe only in the linear-only case (negligible yield); conv factorization without fine-tuning collapses accuracy (≈10%).

### 2.2 Comparative Findings
| Technique | Size | Speed | Accuracy | Verdict |
|---|---|---|---|---|
| Dynamic int8 PTQ | ✅ −28.8% | ~neutral alone | ✅ lossless | **Core stage of Pipeline G** |
| Unstructured pruning (40%) | ✅ once folded | ✅ once folded | ✅ neutral | **Foundation stage of Pipeline G** |
| TorchScript graph opt | ✅ strong | ✅ strong | ✅ lossless | **Decisive stage of Pipeline G** |
| Distillation (pretrained) | ✅ strong | ~neutral | ✅ strong | Best FX-path core (F/F2), but speed-bound |
| `torch_fx` fusion | neutral | neutral | ✅ lossless | Always-on, but not enough for speed |
| Static PTQ / QAT | ✅ strong | ❌ / neutral | ❌ collapses / borderline | Excluded from ship path |
| Low-rank | weak | weak | mixed | Optional only |

**Reading the results:** no single technique meets all three CTO targets, but the **combination** in Pipeline G does — unstructured pruning as an accuracy-neutral base, dynamic int8 for lossless compression, and TorchScript to fold both into a small, fast, frozen graph. Quantization is usable here after all, provided it is **dynamic** (not static) and paired with graph optimization.

## 3. Multi-Stage Compression Pipeline

### 3.1 Final Pipeline Choice
The recommended pipeline is **G**, applied to the baseline MobileNetV3-Small backbone:
1. **40% global unstructured pruning** — accuracy-neutral base that zeroes low-magnitude weights,
2. **dynamic int8 PTQ** on the `nn.Linear` layers — lossless compression,
3. **TorchScript graph optimization** — `torch.jit.trace` → `torch.jit.freeze` → `torch.jit.optimize_for_inference`, which folds the pruned zeros, inlines parameters, and fuses Conv+BN into a frozen inference-only artifact.

This is the only pipeline that clears all three CTO targets simultaneously. It uses **dynamic** (not static) int8 to avoid the MobileNetV3 accuracy collapse, and TorchScript (not `torch_fx`) to supply the size and speed the FX pipelines lacked.

### 3.2 Pipeline Results
| Metric | Baseline | Pipeline G | Result vs target |
|--------|----------|-----------|------------------|
| Model size (MB) | 5.96 | **0.59** | **−90.2%** (target −30% ✅) |
| CPU speedup | 1.00× | **2.0×** | **past 1.67×** ✅ |
| Top-1 accuracy (%) | 88.40 | **85.9** | **−2.5 pts** (floor 83.98% ✅) |

Pipeline G is the first and only configuration that clears **all three** goals at once. For context, the full ranking from Notebook 03:

| Pipeline | Size | Size Δ | CPU speedup | Top-1 | Acc Δ | All 3? |
|---|---|---|---|---|---|---|
| **G** — prune 40% → dynamic int8 → **TorchScript** | **0.59 MB** | −90.2% | **2.0×** | **85.9%** | −2.5% | **PASS** |
| F2 — distill(pretrained w1.0,160,head128) → fx | 3.95 MB | −33.7% | 1.25× | 89.7% | +1.3% | NO (speed) |
| F — distill(pretrained w1.0,160,head256) → fx | 4.24 MB | −28.9% | 1.23× | 91.0% | +2.6% | NO |
| A — prune → fx → dynamic int8 | 4.13 MB | −30.6% | 1.07× | 86.9% | −1.5% | NO (speed) |
| D — distill(w0.75,224) → fx | 2.65 MB | −55.4% | 1.13× | 75.3% | −13.1% | NO |
| E@128 — distill(w0.85,128) → fx | 3.16 MB | −47.0% | 1.16× | 71.6% | −16.8% | NO |
| E@112 — distill(w0.85,112) → fx | 3.16 MB | −47.0% | 1.44× | 68.4% | −20.0% | NO |
| B — distill → fx → dynamic int8 | 1.80 MB | −69.7% | 1.33× | 63.2% | −25.2% | NO |
| C — distill → prune → fx → dynamic int8 | 1.80 MB | −69.7% | 1.41× | 58.8% | −29.6% | NO |

Every FX-only pipeline satisfies at most two of three: the pretrained students (F, F2) solve accuracy and size, but **none reaches the speed target** because FX fusion alone does not restructure the graph. TorchScript is what closes the gap.

### 3.3 Why This Pipeline Wins
- **Pruning 40% (unstructured)** is an accuracy-neutral foundation — it zeroes low-magnitude weights with no retraining, and gives its size/latency payoff once the graph runtime folds the zeros in.
- **Dynamic int8 PTQ** is lossless on this model (unlike static PTQ), compressing the Linear layers without touching accuracy.
- **TorchScript graph optimization** is the decisive lever: tracing + freezing + `optimize_for_inference` inlines parameters, folds constants, and fuses Conv+BN into a frozen artifact — delivering both the size and the 2.0× speed the FX pipelines could not.
- **Static quantization stays excluded** because it collapses MobileNetV3's hard-swish / SE ranges; only the dynamic variant is used.

## 4. Mobile Deployment Findings

### 4.1 Preprocessing Fix Already Implemented
The earlier issue was a fragile in-graph upsample. It is resolved by moving the resize into preprocessing, so the mobile export no longer depends on a model-internal interpolation path. The deployment notebook sets `DEPLOY_RESIZE = 224` — Pipeline G is the baseline backbone traced at its native resolution — and the image transforms resize before the model sees the input.

This matters because Pipeline G is a **traced** TorchScript graph: with resize in preprocessing the backbone is `F.interpolate`-free, so tracing is safe and `optimize_for_mobile` can take its fastest XNNPACK path without silently producing a numerically broken graph.

### 4.2 Mobile Export Validation
Pipeline G's final stage is a frozen, inference-optimized int8 TorchScript `ScriptModule`. That graph serializes with `torch.jit.save` but does not reliably survive `torch.jit.load` on this build, so the deployment notebook **rebuilds G in-memory from its recipe** (baseline → prune 40% → dynamic int8 → TorchScript) rather than reloading a checkpoint. The export is then guarded by a **real-data parity gate**: it tries the fastest conversion first (`optimize_for_mobile` / XNNPACK) and keeps the first strategy whose top-1 predictions match the source model on a real labeled batch, so a numerically broken bundle can never be saved. This replaces the earlier single-random-tensor check that could report a false PASS.

### 4.3 Current Mobile Conversion Results
| Metric | Pipeline G (source, measured nb 03) | Mobile bundle (`.pt`) |
|--------|-------------------------------------|-----------------------|
| On-disk size | **0.59 MB** (−90.2% vs 5.96 MB) | ≈ source; `optimize_for_mobile` adds only prepack metadata |
| Top-1 accuracy | **85.9%** (−2.5 pts) | parity-gated to within 1 pt of source |
| CPU speedup vs baseline | **2.0×** (≥ 1.67×) | ≥ source on ARM once XNNPACK conv prepack applies |

This workspace is x86, so re-running selects the unified `x86` engine and reports desktop-x86 milliseconds; the actual mobile target is ARM (`qnnpack`), which differs in absolute time, but the recipe (dynamic int8 + TorchScript) is what generalizes. The parity gate enforces accuracy equivalence between the mobile bundle and the source; real-device p50/p95/p99 latency still needs on-ARM measurement.

### 4.4 Production Considerations
- **Preprocessing consistency** is controlled but must stay versioned across training and mobile inference (the resize/normalization contract).
- **ARM benchmarking remains mandatory** — x86 timing is not representative; validate the `qnnpack` engine and p50/p95/p99 latency on the target device tier.
- **Static quantization stays excluded**; the shipped path uses **dynamic** int8 only, until a per-module fp32 qconfig for the SE/hard-swish blocks is validated.
- **Parity checking should remain in CI** so broken exports are caught before shipping.

## 5. Conclusion
All three CTO targets are met by **Pipeline G** (prune 40% → dynamic int8 → TorchScript):
- **size 0.59 MB (−90.2%)**, well under the 4.17 MB / −30% bar,
- **2.0× CPU speedup**, past the 1.67× / −40% bar,
- **85.9% top-1 (−2.5 pts)**, comfortably inside the 83.98% floor.

The key insight is that the earlier "speed is an unreachable wall" conclusion was wrong: it held only for eager-mode `torch_fx` pipelines. **TorchScript graph optimization** — tracing to a frozen, Conv+BN-fused, inference-only graph that also folds the pruned zeros — was the decisive size *and* speed lever. Quantization is usable here after all, provided it is **dynamic** (lossless) rather than static (collapses). The resize-in-preprocessing fix and the real-data parity gate make the traced mobile export safe and verifiable.

## 6. Recommendations for the Next Step
1. **Benchmark the Pipeline G mobile bundle on real ARM** (Android/iOS) to confirm the 2.0× CPU speedup and on-device accuracy parity that the gate already enforces on desktop.
2. **Add light structured channel pruning before TorchScript** to remove real channels/FLOPs and bank extra latency/size headroom above the CTO thresholds.
3. **Recover accuracy headroom on G** (currently 85.9%) with a short fine-tune or QAT pass after pruning, buying margin above the 83.98% floor.
4. **Keep the preprocessing contract versioned** and **preserve the real-data parity gate in CI** to prevent silent regressions.

## 7. References
- Howard et al., *Searching for MobileNetV3* (ICCV 2019).
- Hinton, Vinyals, Dean, *Distilling the Knowledge in a Neural Network* (2015).
- Zhu & Gupta, *To Prune, or Not to Prune* (2017) — gradual (cubic) pruning schedule.
- PyTorch documentation: [Quantization](https://pytorch.org/docs/stable/quantization.html), [torch.fx](https://pytorch.org/docs/stable/fx.html), [Mobile / optimize_for_mobile](https://pytorch.org/docs/stable/mobile_optimizer.html), [torch.linalg.svd](https://pytorch.org/docs/stable/generated/torch.linalg.svd.html).
