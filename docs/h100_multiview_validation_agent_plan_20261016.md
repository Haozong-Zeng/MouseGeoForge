# MouseGeoForge H100 multiview validation plan and agent handoff

## Current execution override — 2026-10-19: diagnose, prove learnability, then pair

**This section supersedes every execution order, data-admission suggestion and
allocation schedule below. The remainder is dated historical context.** This
update prepares source and an H100 handoff; it starts no training, rendering,
cluster allocation, automation, sealed scoring or keypoint run. The completed
local v3 cycle is closed. Keep `v3-12` and `pymousestudio` paused.

Read the [reconciled route](bottom_geometry_v3_reconciled_plan_20261018.md#7-合并-claude-四臂审核事实修订与更小的决策顺序)
and [compact evidence](experiments/bottom_geometry_v4_handoff_20261019.json).
Four seed17 arms finished 1,500 updates each; **none passed geometry gates**.
Do not promote A2 or resume an old arm with a new objective. A2 has an additional
implementation defect: the original geodesic teacher gives substantial gradients
at floating-point-sized discrepancies, and its batch preflight omitted teacher
and real anchor. New v4 code is separate; historical v3 source/results stay intact.

### Questions, evidence and stopping conditions

| Stage | Exact question and comparison | Required evidence / stop rule |
| --- | --- | --- |
| E0: zero training | Reevaluate release + four final EMA models under v4; release original / gray-background / blurred-background inputs | Separate genuine joint failures from scale-estimator artifacts; report errors <5°/<15°, descriptive 90°/120°/180° peaks, pair lists and rig-paired differences. No new optimizer updates. |
| L0: bounded learnability probe | A1-SI candidate, camera head LR1e-4, encoder1e-6, depth head1e-5; calibrated common real anchor; teacher OFF; 300 updates, seed17 | Discard probe weights. Tentative preregistered engineering criteria: relative-rotation component decreases ≥30% on the same fixed TRAIN probes and TRAIN bottom rotation median decreases ≥20%; report <15° and peak fractions. Total camera loss alone cannot pass. Failure rejects this recipe, not the information content of the images. Review before a formal pair. |
| P0/P1: minimum recipe pair | Both fresh release, same data/order/effective batch8, same real anchor and fixed numeric geometry weights; **only P1 adds a smooth weak non-bottom teacher** | 3,000 updates/arm, seed17, 500 warmup then cosine, EMA/raw fixed endpoint. Other-view rotation median drift ≤0.5° and raw joint coverage loss ≤2pp are screening criteria. Bottom angular correctness must improve; report paired uncertainty. Do not compare C-without-anchor to T-with-anchor and call it teacher-only. |
| Confirmation / scaling | Only after review of a complete P0/P1 pair: seeds29/43, then separately registered frozen-encoder G0/G1 if warranted; full admitted-pool learning curves | Preserve complete matched arms, sources and seeds. Do not automatically launch controls, more seeds or a larger run because hardware is free. Original strict geometry promotion gates remain in force. |
| Localization | Only accepted geometry advances to frozen-encoder 2D-head convergence, same-head frozen/adapted comparison, then 3D keypoints | Known-camera manual-ray reprojection is not automatic localization or 3D keypoint accuracy. No keypoint head is created by v4. |

The 300-update probe uses a short **50-update warmup** solely to make a bounded
optimization diagnostic feasible. The formal 3,000-update pair uses **500**;
these are different declared endpoints, not a continuation. Reinitialize both
formal arms from release and freeze their complete schedules before either
updates. The displayed weights in the example config are **placeholders** for
teacher/anchor, not a validated recipe. Camera0.049 is not by itself proof of an
insufficient update; inspect module and R/T/FoV gradients and actual angular
learning. AdamW update size cannot be inferred linearly from a loss coefficient.

If E0 supports background symmetry as a substantial contributor, register a
**data revision before** L0/P0/P1: consistently render shared 3D asymmetric
background/fixture layouts and varied lighting, hold out whole rigs/backgrounds,
and cross admitted motions. Do not add camera IDs, pose-encoded colors or
independent per-view background changes as a shortcut. Apply the same revised
data to both arms, freeze a new manifest, and do not simultaneously attribute
teacher gains to a data change. Masking/blur is a distribution-changing diagnostic;
its success supports a hypothesis, its failure does not prove no symmetry issue.

### Evaluation contract v4

- Keep raw predicted-camera + predicted-depth transfer as a primary consistency
  measure. One common positive scale cancels; its observed decline is real.
  Project with positive depth validity, without an absolute-unit denominator
  clamp. Retain invalid predictions in the denominator.
- Fit a **single body-only log-median GT-oracle scale per multiview pack** for
  shape/known-camera diagnosis. Report per-view body-oracle alignment separately
  as shape-only; it cannot establish shared geometry or be used in deployment.
  Log-L1 optimal alignment need not minimize AbsRel/reprojection. Never rank
  differing gauges by unaligned absolute depth or known-camera transfer alone.
- A TRAIN-fitted global real scale is a calibration diagnostic. Match its input
  order to validation; the v3 first-pack mismatch materially biased real scores.
  A known/predicted non-bottom baseline ratio is a candidate annotation-free
  scale estimator, **not a proven fix**. Explicitly count unavailable estimates
  as failures, and report its coverage and failure count.
- Preserve original star-pair SO(3)/direction endpoints, add all-pair results,
  explicit view IDs, GT-angle bins and strict side-side versus all-non-bottom
  labels. Keep median/P95 alongside correctness/peak fractions. The published
  v3 replay86.7° bin included top views; strict side-side was117.8°.
- Other-view rotation drift is a contributor to test, not the unique cause of
  joint regression: A2's aggregate median changed only0.022° while joint coverage
  lost7.93pp. Examine T/Z, direction and K too. The optional `--decompose` reports GT R/center/K/Z swaps on fixed support; swapping R preserves predicted camera centers and all swaps are non-additive oracle diagnostics. Do not treat a universal32px-to-
  rotation conversion or five model-level aggregate points as causal evidence.
- The unseen-rig unit is a **rig**, not six frames. Use matched per-rig differences
  and descriptive rig bootstrap intervals. Real collections share one physical
  rig; they cannot supply independent-rig intervals. Reused development rigs are
  not confirmation. Preserve the8 sealed rigs for a separately reviewed frozen
  policy; none are opened by these tools.

Background interventions preserve a fixed dilated body region with soft edges;
run original/gray/blur on matched contexts and include originally correct source
controls. **Every view needs a verified body mask.** Some real manual packs have
no such mask; v4 reports them as skipped rather than treating empty masks as
permission to erase the animal. Admit a real mask source before running that
branch. Compare only explicitly matched support and report exclusions; do not
let loss of difficult samples masquerade as improvement.

### Exact data scope for this geometry route

Use the full **admitted TRAIN pool**, with balanced mixed samples and source
exposure reporting. A mechanism pilot is not an assertion that the pool was
exhausted or training converged. No Water Reaching in any role of this route.
Admitted crossed renders: ClimbWall and MoSeq,3 frames each,48 TRAIN/12 DEV/8
sealed rigs. CRIM13 is free-motion replay, not a third crossed motion.
PyMouseStudio free-moving real/rendered data may be used with fixed role
manifests. Its nine real calibrations describe one rig: known-camera deployment
and unknown-rig generalization remain distinct tracks.

DANNCE1's full published postprocessed payload contains4,500 times/27,000 views,
but its metadata says **SAM2 calibrated6of6 visual-hull pseudo-depth**, not raw
Gaussian depth or independent GT. Keep confidence≥212,1px erosion, foreground
only and pseudo-geometry weight0.2; all6 cameras/15 pairs/6 anchors remain covered
by the admitted schedule. DANNCE2, GR2/UR2/UR3, protected synthetic TEST and
unadmitted external sources remain excluded here. Earlier historical permission
for other tasks does not silently admit them to this protocol. Keep the recorded
asset-only WalkAround opening and all rejected render/preflight attempts in the
local evidence; no protected TEST model inference occurred in this cycle.

### Source package and commands

New entry points:

- [v4 evaluator](../tools/bottom_geometry_v4_eval_20261019.py): DEV or48 fixed
  TRAIN-rig probes; no sealed option, existing outputs cannot be overwritten.
- [TRAIN-only calibration](../tools/calibrate_bottom_geometry_v4_20261019.py):
  32 group-stratified probes; fixed5°/2% student perturbation; encoder and camera
  budgets separately. Teacher coefficient≤0.05 and≤20% of active geometry median
  budgets at P75 probe norm; real anchor uses the same module rule at100% budget.
  These are initialization budgets, not guarantees for every subsequent sample.
- [v4 runner](../tools/run_bottom_geometry_v4_20261019.py): immutable preparation,
  complete-objective B1/2/4 preflight, explicit P0/P1 training, full-state resume.
- [gradient diagnostics](../tools/diagnose_bottom_geometry_v4_20261019.py): fixed
  TRAIN task/module gradients, R/direction/FoV components and cross-source cosines
  at0/every250/final, with weights frozen throughout an arm.
- [paired report](../tools/compare_bottom_geometry_v4_20261019.py): CPU matched
  support and per-rig bootstrap; no method promotion or follow-up dispatch.
- [disabled probe template](../configs/bottom_geometry_v4_probe_20261019.json),
  [CPU fixtures](../tests/test_bottom_geometry_v4_20261019.py),
  [opt-in GPU check](../tools/check_bottom_geometry_v4_gpu_20261019.py).

The runner does **not** use the old v3 supervisor. It requires an explicit
execution-enabled config, `--execute`, absolute UTC optimization/hard deadlines,
a TRAIN calibration receipt, content-pinned data and a passed preflight. A
formal3,000-update pair additionally requires a reviewed300-update learnability
receipt and both arms' preflights before either formal optimizer starts. Full
state includes raw/EMA/AdamW/buffers/RNG/sample offset and source/data/release
identity. A failure during optimizer/EMA mutation must retain the last complete
checkpoint, not save a partially committed update. Endpoints are evaluated in
separate explicit commands; no control, sealed or keypoint job is dispatched.

A Git clone does **not** include payloads. On nodes separately provide the
versioned v3 `manifest.json`, `data_file_pins.json`, referenced dataset/render
files, release checkpoint, optional four final v3 checkpoints, and the fixed
PyMouseStudio rectified calibration file. Keep repo-relative paths consistent;
use node-local staging/symlinks instead of editing labels or silently replacing
assets. `prepare` and run entry verify **content SHA256/bytes**, not copied mtimes;
this startup cost is part of wall time. The8 sealed payloads may be checked as
files, never scored by the model. Upstream source used locally is
`facebookresearch/vggt-omega` commit `39a0cb8af88554f15ddcb5354cd52bde588fa014`
with project modifications. **This repository already tracks the model and
training-helper Python sources under `vggt-omega/`; use this parent repository's
versions, do not overwrite them with a pristine upstream clone.** Provide the
pinned release separately. Use Python3.11+ (the project helpers use `tomllib`),
a node-compatible PyTorch/CUDA runtime, and install the local package plus helper
dependencies, e.g. `python -m pip install -e ./vggt-omega pandas matplotlib scipy`.
CPU fixtures do not validate a different GPU/kernel/runtime stack; repeat full
preflight on H100. The new protocol pins actual model AND training-helper Python
sources too. Do not execute historical helper-script main functions.

Illustrative Bash commands from the repository root; none have been scheduled:

```bash
V4_PY=.venv-vggt-omega-eval/bin/python
V4_MANIFEST=output/bottom_geometry_v3_local12h_20261018/manifest.json
V4_RELEASE=vggt-omega/vggt_omega_1b_512.pt
V4_CONFIG=configs/bottom_geometry_v4_probe_20261019.json

$V4_PY tests/test_bottom_geometry_v4_20261019.py
$V4_PY tools/bottom_geometry_v4_eval_20261019.py --manifest "$V4_MANIFEST" --release "$V4_RELEASE" --decompose --output output/v4_E0/release.json
# Repeat explicitly for each v3 latest.pt with --checkpoint ... --state ema.
# No default loop or automatic model selection.
$V4_PY tools/bottom_geometry_v4_eval_20261019.py --manifest "$V4_MANIFEST" --release "$V4_RELEASE" --background gray --sources unseen_rig_gate free_motion_replay --output output/v4_E0/release_gray.json
# Repeat the exact same selection with --background blur.

$V4_PY tools/calibrate_bottom_geometry_v4_20261019.py --config "$V4_CONFIG" --output output/v4_E0/calibration.json
# Copy the template to a new config; insert calibrated weights/receipt, reviewed
# stage and deadlines. Keep execution_authorized=false until training is authorized.
$V4_PY tools/run_bottom_geometry_v4_20261019.py prepare --config path/to/frozen_config.json
$V4_PY tools/run_bottom_geometry_v4_20261019.py preflight --config path/to/frozen_config.json --arm P0
$V4_PY tools/run_bottom_geometry_v4_20261019.py preflight --config path/to/frozen_config.json --arm P1
# Only after separate authorization; a disabled config deliberately refuses this:
$V4_PY tools/run_bottom_geometry_v4_20261019.py train --config path/to/frozen_config.json --arm P0 --execute
# --resume restores the same complete state and immutable objective, not a revision.

# After a fixed endpoint, evaluate EMA AND raw in distinct, fresh output files:
$V4_PY tools/bottom_geometry_v4_eval_20261019.py --manifest "$V4_MANIFEST" --release "$V4_RELEASE" --checkpoint path/to/P0/latest.pt --state ema --train-probes --output output/v4_E0/P0_train_probes.json
$V4_PY tools/bottom_geometry_v4_eval_20261019.py --manifest "$V4_MANIFEST" --release "$V4_RELEASE" --checkpoint path/to/P0/latest.pt --state ema --output output/v4_E0/P0_dev.json
$V4_PY tools/compare_bottom_geometry_v4_20261019.py --reference output/v4_E0/release.json --treatment output/v4_E0/P0_dev.json --output output/v4_E0/P0_delta.json
```

For a formal pair create a new namespace/config with3,000 updates,500 warmup,
seed17, the **same** frozen calibrated weights for both arms and a reviewed
`learnability_receipt`. Receipt fields: `passed`, `optimizer_steps=300`,
`manifest_sha256`, `release_sha256`, `r_component_reduction≥0.30`,
`train_bottom_median_reduction≥0.20`, plus links/hashes of fixed TRAIN probe and
R-component evidence. It is a reviewed engineering decision, not an automatic
claim of generalization. Do not reuse the L0 checkpoint as a formal start.

### H100 allocation and engineering readiness

Prefer completing and reviewing E0/L0 locally before booking the cluster; if
the allocation arrives first, they remain bounded prerequisites, not permission
to fill the window with failed recipes.

Four H100s for48h is a planning ceiling (192 GPU-hours), not an active booking
or a target to exhaust. Confirm topology, VRAM, scheduler, storage and access
first. Prefer **one GPU per independent arm/seed**; do not implement DDP merely
to fill devices or increase effective batch. Preserve effective batch8, sample
order and per-sample reductions. H100 may allow larger physical batches, but
encoder batching remains sample-wise until independently passing numerical
parity. Benchmark end-to-end loading/backward/checkpoint/evaluation cost, not
just utilization. Local1,500-update costs were55–79min; a3,000-update arm is
provisionally2–3h locally, **not an H100 estimate**.

| Allocation-relative time | Conditional work |
| --- | --- |
| 0–4h | Payload/environment audit, CPU fixtures, content hashes, full state and B1/2/4 GPU checks, throughput measurement |
| 4–10h | E0: independent zero-training evaluations/background controls; review whether data revision is needed |
| 10–14h | L0 only if separately authorized; otherwise stop. On failed learnability, close this recipe rather than consume the remaining allocation |
| 14–30h | Complete P0/P1 seed17 pair only after E0/L0 review; fresh common release, no treatment-only run |
| 30–40h | Reserve for reviewed confirmation/extra seeds or necessary engineering recovery; no automatic G0/G1 or keypoints |
| 40–48h | Optimization off; final raw/EMA evaluation, paired/source reports, closure and resource release. Sealed scoring still requires its own reviewed frozen policy |

Use an absolute scheduler hard limit and earlier optimization cutoff on every
job; start time and stage decisions must be registered after allocation.
Within-node independent jobs can use the per-physical-GPU lock: set
`CUDA_VISIBLE_DEVICES` to one physical `GPU-...` UUID and set
`MOUSEGEOFORGE_GPU_LOCK_ID` to **the identical UUID**. Without this explicit
binding tools retain the original global local lock and serialize safely.
Record node/job ID/GPU UUID/PID/start identity before process control; do not
start duplicates. Existing old automations are not a cluster scheduler.

Validated locally on2026-10-19: CPU mathematical/state fixtures and GPU complete
P0/P1 B1/2/4 parity, including existing adapted weights. Largest gradient relative
L2 error was0.000083 (0.0083%); loss relative error≤1.20e-7. The old A2 counterexample
was6.793 (679%). Teacher self-retention loss is now0 in the matched forward path.
These are **implementation tests with zero model optimizer updates**, not new
training, H100 throughput, E0 background results or proof of a successful recipe.
The full-model optimization/resume loop has not been exercised with new VGGT
updates; CPU AdamW next-update restoration and no-update GPU gradients were
verified. Recheck finite updates and complete-state restoration on a disposable,
separately authorized engineering state before any formal endpoint.

---

## Historical proposals below — not the current execution queue


## Bottom-geometry mechanism update — 2026-10-18

Use the [reconciled v3 scale/retention plan](bottom_geometry_v3_reconciled_plan_20261018.md)
for the local geometry experiment order: zero-training scale diagnostics,
native-GT versus shared-scale recipe screening, retention, then matched G0/G1
encoder comparisons. The initial encoder-by-projection four-arm design is a
historical alternative. No new training, rendering or queued control has started.

The full admitted training-pool, external2D and H100 requirements below remain
in force. The local1500/4000-update budgets and proposed48/12/8 rigs are mechanism
pilots, not limits on full-data training or evidence of convergence. Frozen
released-encoder external2D readout tests remain separate from promotion of
adapted geometry. [Zero-training scale evidence](experiments/bottom_geometry_zero_train_scale_20261018.json)
distinguishes training-fitted calibration from validation-label oracle diagnosis;
[head/gradient diagnostics](experiments/bottom_geometry_failure_diagnosis_20261018.json)
do not establish a single cause or promote a checkpoint.

## Latest planning update — 2026-10-18 data scale and external bottom views

Use [the current strategy and data-scale plan](data_scale_and_current_strategy_20261018.md)
for the latest data and experiment priorities. H100/multi-GPU work should
prioritize a sufficiently trained **full admitted training pool**, with
acquisition/rig-based 25/50/100% learning curves where budget permits. Small
subsets are implementation pilots, not the default allocation workload.
Repeated views or adjacent video frames are not independent animals, rigs or
captures; monitor exposure and held-out generalization, not GPU utilization alone.
"All data" excludes development/sealed tests, unadmitted data and unsupported
supervision. Measure multi-GPU throughput and preserve complete matched arms
and seeds17/29/43; more data/GPUs are not a guarantee against overfitting.

The latest external review favors PanoMouse as a conditional 2D bottom/paw
supervision source. Its released3D reprojection median/mean is9.48/12.95 native
pixels across92 landmarks, not calibration-board error or physical3D accuracy.
Its PKL and JSON contain identical K/R/T, with no calibration residual report.
OSF remains pending collaborator calibration/physical-dimension/annotation and
license evidence. Neither dataset is silently admitted for training.

Known-calibration deployment and unknown-rig camera generalization are separate
questions. A frozen released-encoder external2D heatmap baseline is an independent
data/readout test; it does not promote the failed geometry checkpoint. Adapted
geometry-to-keypoint progression remains gated. The scale/retention diagnosis,
data expansion, paired controls and resource rules are specified in the linked
strategy. This documentation update starts no training, cluster allocation or
automation; dated execution claims below are historical.

## Current priority override — 2026-10-18 final geometry review

The [bottom-view geometry roadmap](bottom_geometry_experiment_roadmap_20261018.md)
is the current execution order; dated priorities below are historical proposals.
The pure-geometry feasibility run completed 4,000 updates, including final raw
and EMA validation. It did not pass unseen-rig bottom pose or other-view
retention gates. Known-calibration real reprojection gains do not establish
unknown-device camera generalization or automatic 3D keypoint accuracy.

Diagnose geometry and retention, register any recipe/data revision, then execute
a matched geometry pair from common released weights. Only accepted geometry
can advance to frozen-encoder 2D-head convergence, a same-head frozen/adapted
comparison, and finally 3D keypoints. The [paired v2 protocol](bottom_domain_paired_v2_20261018.md)
is registered, not executed. This feasibility run is not retroactively a paired
arm. All current supervisors exited with automatic next-arm dispatch disabled;
no frozen control, keypoint job or H100 allocation is launched by this update.
See [final raw/EMA evidence](experiments/bottom_geometry_v2_final_20261018.json).

**Latest 2026-10-18 correction:** the joint geometry/2D run below was saved and stopped at step13,568 by user instruction. The active plan is [pure geometry bottom adaptation v2](pure_geometry_bottom_20261018.md): fresh release, no keypoint head, mixed8-micro updates with25% corrected real correspondence supervision and0% duplicate-rig video camera updates, measured task gradients, geodesic rotation, warmup/cosine and EMA. All nine PyMouseStudio calibrations are identical; known-rig predicted camera scores are diagnostic only. Entire newly rendered rigs are held out for camera/generalization gates. The old queued control is cancelled; closed cycles/automations remain closed.

**2026-10-18 authorized priority update:** bottom-view foundation domain adaptation now precedes new keypoint architecture/3D optimization. The initiating evidence is the user's direct inspection of released-demo outputs on real multiview videos including bottom views. See [the active bottom adaptation plan](bottom_domain_adaptation_20261018.md): all admitted synthetic/real training pools, acquisition/motion-source held-out validation, side/top replay, fresh release initialization and a matched frozen-encoder control. Validate camera/depth geometry and bottom 2D localization before any 3D-keypoint promotion. Local training is authorized and started; H100 remains a future option. Historical results and closed-cycle stop markers below remain historical.

Created: 2026-10-16. Updated: 2026-10-17 after the local cycle closeout and the
July Water Reaching versus October 6 free-behavior implementation audit.
Priorities revised again on 2026-10-17 after an external plan review
([experiment_plan_review_20261017.md](experiment_plan_review_20261017.md)).
Status: the user subsequently authorized adopting this review and beginning
local execution on 2026-10-17. The active, bounded P0 work and standing benchmark
are defined in [local_validation_plan.md](local_validation_plan.md). The H100
allocation has not started. Both preceding local cycles remain closed, and
the explicitly paused automation remains paused.

**Revised priority (2026-10-17).** The frozen original five-point selector
pipeline already meets the stated quality target on FR3 CAL in all three seeds
(coverage32 97.27%, P95 23.06/21.02/23.44 reference pixels). CAL is a
repeatedly developed single acquisition, so this is optimistic evidence, not
proof. The first question is therefore no longer which single-view head to
build, but **whether the existing best system holds on unseen acquisitions**.

**P0 sealed result (2026-10-18):** on GR2/UR2/UR3 (75 groups, 475 views, 1,697
known points) the frozen selector scored coverage32 94.40/94.46/95.88% and P95
40.1/36.4/28.8 reference pixels; the deployment gate passed for 1 of 3 seeds.
The tail is concentrated in bottom camera A and the left hindpaw. The selector
still beat its fixed-entropy control by 1.7–2.0 pp. GR2/UR2/UR3 have now been
looked at once and must not be used for method selection. Details:
`output/validation_benchmark/runs/p0b_sealed_20261018_v1/report.md`.

The order is now:

1. **P0 — validate and measure (local).** Independent sealed evaluation of the
   frozen selector pipeline on GR2/UR2/UR3; a standing leave-one-session-out
   development benchmark with effect-size gates; a Lightning Pose baseline on
   the same protocol.
2. **P1 — attack identity structurally (local).** Chirality-aware 3D paw
   identity assignment in a body frame; the H0/H1/H2 heatmap bridge plus a
   system-level July-recipe transplant arm (H3). These fit on the local GPU.
3. **P2 — scale (H100).** Multiview-consistent pseudo-label self-training on
   unlabeled real video and multiview end-to-end VGGT adaptation, with DANNCE,
   SBeA and Orbbec as admitted training sources.
4. **P3 — temporal (H100 remaining budget).** Lightning-Pose-style temporal
   context and EKS as a controlled contrast on top of the best P2 system.

The user has now permitted use of the locked GR2/UR2/UR3 data, internal
training use of real pseudo-labels, and temporal modeling. These permissions
change scope, not the evidence standard: each still requires manifests, role
assignment and exposure records. Pseudo-labels remain internal; publication or
release is still not authorized. This document authorizes no new run,
supervisor, queue or allocation by itself.

The user expects at least four NVIDIA H100 GPUs, provisionally 80 GB each, for
48 continuous hours. Four GPUs provide a nominal ceiling of 192 GPU-hours;
initialization, data loading, evaluation and failures consume part of that
budget. Node topology, interconnect, storage, scheduler and actual availability
are still unspecified. The user permits larger and more diverse datasets,
subject to the admission and exposure requirements below. The October 16
authorization applied to one bounded local cycle, which is now closed; it does
not allocate the H100 cluster or extend either completed local cycle. The
current local execution authorization does not allocate that cluster.

## First actions for the receiving agent

1. Read this document and applicable repository instructions. Inspect the Git
   revision and working tree before changing files.
2. Read the July/October comparison and the completed local cycle closeouts.
   Preserve STOP markers, frozen protocols, outputs and negative results. Do not
   restart either cycle or interpret this document as permission to train.
3. Inventory the datasets, annotations, calibration and upstream checkpoints
   actually available on the new nodes. Distinguish a synchronized multiview
   time point from an acquisition session, an animal and an image.
4. Prepare versioned manifests for training, development and sealed testing.
   Record historical exposure, including exposure through upstream detectors.
5. Admit GR2/UR2/UR3 as **sealed test** data: build their manifests, map their
   labels to the five-point schema and record upstream exposure, without
   opening predictions or scores until the frozen selector pipeline is fixed.
6. Complete the read-only error decomposition on admitted, exported predictions.
   Separate anatomical identity, spatial localization and instance detection;
   keep GT-assisted diagnostics out of the automatic endpoint.
7. If the user later authorizes execution, run P0 and P1 on the local GPU
   first (see "Revised priority order"). Reserve the H100 allocation for P2/P3
   workloads that need it. Resolve allocation, absolute start/stop times,
   ownership, endpoint budgets and gates in a new configuration. The schedule
   below is relative to that future start, not an active authorization.

Repository source, local research evidence and execution readiness are distinct.
The repository is source-only. Many recent scripts and all experiment outputs,
datasets and checkpoints may exist only in the original workspace. A GitHub
clone alone must not be assumed to contain a runnable reproduction. Record
missing dependencies and arrange their separate, authorized availability;
do not invent missing results or silently substitute a different checkpoint.

## Evidence from the completed 18 hour round

The previous round was `semantic_matching_cycle_20261015`. Its authorized window
was 2026-10-15 14:00:35 UTC to 2026-10-16 08:00:35 UTC, with optimization ending
at 07:30:35 UTC. Its recorded final status is
`closed_frozen_automation_paused`. The existing `pymousestudio` automation was
paused. Do not restart that supervisor or append to that queue.

The old corrected manifest contained 1,202 FIT/CAL image rows:

| Partition | Synchronized real groups | Image views | Known five-point annotations |
| --- | ---: | ---: | ---: |
| Real FIT | 134 | 874 | 3,085 |
| Real FR3 CAL | 22 | 154 | 573 |
| Synthetic FIT | Not expressed as real groups | 118 | See its annotation manifest |
| Synthetic CAL | Not expressed as real groups | 56 | See its annotation manifest |

The evaluated points were nose tip 1 and paws 15, 16, 17 and 18. The real FIT
paw-only denominator was 2,432 known annotations. These counts describe the old
round, not the proposed expanded dataset or the number of independent animals.

The original selector reference uses its frozen 14-feature schema, L2 = 0.01,
fixed Newton fitting and selection probability greater than 0.5. Preserve its
standardization and seed-specific states when replaying it. A selector refitted
on expanded training data is a new baseline and must be named separately from
the archived model.

| Observation | Recorded result | Consequence for this plan |
| --- | --- | --- |
| Frozen original five-point selector | Mean CAL coverage within 32 input pixels was 97.266%, versus 95.230% for its control; P95 was 23.06, 21.02 and 23.44 pixels across seeds 17, 29 and 43 | Retain it as a reference and reproduce it before comparing new models |
| Sparse source plus target fusion error decomposition | 191-197 ranking misses versus 27-43 pool misses per seed, out of 2,432 known paws | Prioritize choosing the correct semantic candidate |
| Identity error proxy | 82-88% of ranking misses were near another known paw; contralateral paws were the largest category | Test learned identity correspondence and difficult negatives |
| Common 3D candidate oracle | 2,417/2,419/2,418 of 2,432 annotations were within 32 pixels; P95 was 21.37/21.14/20.78 pixels | The registered candidate pool has useful headroom if selection improves |
| Soft geometry and triangle consistency | Meaningful relative coverage gains, but P95 remained roughly 60-100 pixels in these variants | Geometry helps, but did not solve the severe error tail |
| Pixel occlusion and early fusion | Clean multiview fusion gained about 0.42 percentage points with an interval crossing zero; the pixel augmentation arms hurt performance | Revisit only under a distinct data or representation hypothesis |
| Peer-view ranking objective | Pairwise ordering improved, but final correct choices did not; endpoint coverage fell about 0.38 percentage points | Do not use ranking accuracy alone as a continuation criterion |
| Targeted synthetic descriptor training | About +0.23 percentage points versus random synthetic training, below its registered margin, and worse than the extra-real control | Require richer supervision and matched exposure before expanding this branch |

These observations have different controls and denominators; their gains must
not be added or ranked as if they came from one common benchmark. CAL was a
repeated development set with historical detector FR3/ClimbWall exposure. The
existing out-of-fold meta-validation was not fully nested validation of the
entire pipeline. Group-paired intervals were descriptive.

The candidate oracle uses ground truth and is a feasibility diagnostic, not a
deployable predictor or a measurement of true 3D accuracy. Proximity to another
paw is an error proxy: real paws can overlap, and missing annotations do not
prove absence. The 525 completed technical jobs in the old cycle were not 525
independent experiments.

## Research questions and priorities

The next controlled work, if authorized, should answer, in this order:

1. Does the frozen original selector pipeline, which passes the quality target
   on reused CAL, still pass on unseen acquisitions (GR2/UR2/UR3)?
2. Is the remaining contralateral-paw confusion resolved by an identity
   assignment that can distinguish a pose from its mirror image?
3. Does more independent real data, via multiview-consistent pseudo-labels and
   admitted external datasets, move held-out performance by a step, not by
   fractions of a percentage point?
4. Does temporal context further reduce identity swaps and severe tail errors?

The single-view heatmap questions (stronger readout; bypassing the old
compression interface) remain useful, but they are now P1 local experiments,
not the purpose of the H100 allocation. Do not spend the allocation on a broad
loss, temperature, resolution or head-size sweep.

## Revised priority order (2026-10-17)

### Why the order changed

| Observation | Source | Consequence |
| --- | --- | --- |
| Frozen selector passes coverage32 ≥ 90% and P95 ≤ 32 px on CAL in all three seeds | `semantic_matching_cycle_20261015/final_closeout_v1.json` | Validate the best existing system before building a new one |
| CAL has 22 groups and 573 known points from one acquisition; 0.2–0.5 pp gates equal about 1–3 points, while single-acquisition differences reach ±3.5 pp | DDMS and paw-identity gate reports | Many "gate failed" outcomes were below measurement resolution; change the benchmark and gates |
| Largest within-project gains came from data: extra real updates +2.7 pp; synthetic scenes 64→400 raised held-out PCK 9%→58% | `paw_identity_cycle_20261013.md`, `pose_diverse_data_control.md` | Data scale is the binding constraint; real FIT is only 134 groups from 6 acquisitions |
| Soft radial distances, triangle consistency and bone lengths are mirror-invariant | `paw_identity_cycle_20261013.md`, semantic matching decisions | They cannot in principle correct left/right swaps, the dominant error |
| The July success combined multiview joint encoding, a geometry-adapted checkpoint, a large head and 3D losses | July/October audit table below | Single-factor additions to a failing system can each show ~0 effect when the factors interact |
| A 4000-step head endpoint costs 5–15 GPU-minutes locally | Cycle engineering records | H0/H1/H2 need under 2 GPU-hours; they do not need the H100 allocation |
| Static LP three-model median 24 px versus DLC 44 px; LP temporal reduced >100 px errors 102→73 | `research_review_20261006.md` | Strong external baseline never compared on the five-point protocol; temporal context targets identity swaps |

### P0 Validate and measure (local, before H100)

1. **Sealed independent test of the frozen selector.** Replay the archived
   selector pipeline (extra-real 6000 heads, fixed semantic-contrast geometry,
   original 14-feature selector, all three seeds) exactly, freeze all outputs,
   then score once on GR2/UR2/UR3. Report coverage8/16/32, nearest-rank P95/P99
   with failures as infinite error, per acquisition, per camera and per paw.
   Record that the detector and VGGT checkpoint were frozen; disclose any
   upstream exposure of these acquisitions. No tuning after this score is
   opened; GR2/UR2/UR3 then become reported test evidence, and further method
   development must not select against them.
2. **Standing leave-one-session-out (LOSO) development benchmark.** Seven folds
   over the six FIT acquisitions plus FR3. One frozen scoring harness, one
   result table across cycles, with every new method as a new row. This
   replaces single-CAL decisions and ends the "results across cycles are not
   comparable" problem.
3. **Effect-size gates.** Continuation requires a practically meaningful change
   on LOSO (for example ≥ 3 pp coverage32 or ≥ 30% P95 reduction, with no
   acquisition harmed by more than its fold noise), registered before running.
   Sub-percentage-point gates are retired as continuation criteria.
4. **External baselines on the same protocol.** Lightning Pose multiview
   (static, then temporal in P3) and, if feasible, DANNCE on the same LOSO
   folds and sealed test, with the same five points and units.

### P1 Structural identity and the readout bridge (local)

1. **Chirality-aware 3D identity assignment.** Detect paw candidates without
   left/right labels (or pool contralateral channels), triangulate, and build
   an inference-available body frame from nose, body axis and the vertical
   direction. Assign left/right by the sign of the lateral coordinate and
   fore/hind by position along the body axis, with an explicit unassigned
   state. This is anatomical handedness in 3D, not the prohibited fixed
   image-left rule. Compare against the frozen selector on LOSO; the candidate
   oracle (2,417/2,432 within 32 px) shows the candidates usually exist.
2. **Heatmap bridge.** H0/H1/H2 were completed on 2026-10-17 under the old
   three-fold protocol ([results](heatmap_bridge_cycle_20261017.md)): H1 beat
   H0 by +3.24 pp mean coverage32 in all three seeds with fewer swap proxies;
   H2 showed no consistent gain. Their P95 gate could not pass for any arm
   because 11.2% of points were rejected by the shared detector gate. Next:
   (a) put the H1 head into the original selector pipeline and compare it with
   the extra-real6000 head on 7-fold LOSO under one detector policy;
   (b) H3, a system-level July-recipe transplant (multiview joint encoding,
   final-layer features, July head, 3D-consistency losses). If H3 works,
   ablate it downward instead of building upward from H0.
3. **Simple single-mouse localization.** For fixed-rig single-mouse data,
   compare the six-query detector against a mask/background-subtraction crop
   with top-down rescaling. The detector's top-box IoU≥0.5 rate fell from
   90.7% to 12.8% under 180° rotation, and real bottom bodies are ~2.6× the
   synthetic scale; a normalized crop addresses both.

### P2 Scale with real data (H100)

1. **Multiview-consistent pseudo-label self-training.** Run the best P0/P1
   system over unlabeled synchronized real video (existing rigs, SBeA, Orbbec
   where calibrated). Keep a pseudo-label only when its triangulation is
   supported by enough views with low reprojection residual and agreement
   across seeds; manual-label triangulation reached 96.7% coverage32, so
   consistent multiview 2D is a credible teacher. Pseudo-labels are training
   data only, never evaluation labels, never mixed into LOSO held-out folds or
   the sealed test, and not released.
2. **External labeled data.** Admit DANNCE mouse (3D labels) and other mapped
   sources as training partitions with explicit semantic mapping; evaluate them
   separately.
3. **Encoder adaptation at scale.** LoRA or partial fine-tuning of VGGT with
   multiview joint encoding, real + pseudo-labeled + synthetic data. This is
   the workload that justifies four H100s.
4. Report learning curves by number of independent acquisitions, not frames.

### P3 Temporal context (H100 remaining budget)

Add Lightning-Pose-style temporal context and EKS on the best P2 system as a
controlled on/off contrast with the same split, measuring identity-swap counts
and >100 px errors. Keep neighbouring frames of an acquisition in one role.
Temporal outputs may feed P2 pseudo-label filtering only after the contrast
shows they reduce swaps on LOSO.

## July Water Reaching versus October 6 free-behavior audit

The located July success is a VGGT-Omega multiview heatmap experiment, not an
isolated native-DINOv2/v3 baseline. It demonstrates a useful spatial readout of
these features in that task; it does not compare backbone families. The
October 6 implementation also predicts dense spatial heatmaps. Do not conflate
it with the later fixed-candidate matcher experiments.

| Component | July 28 Water Reaching | October 6 free behavior | Interpretation |
| --- | --- | --- | --- |
| Encoder weights | Geometry-adapted step6000 checkpoint, frozen during heatmap-head training | Released VGGT checkpoint, frozen | Frozen does not mean the same initialization |
| Feature interface | Final aggregation-layer tokens, 2048 channels, with a trainable input projection in the new head | Aggregation layers 4/11/17/23, each projected to 256 channels, averaged, then sine position encoding added | October uses learned compressed instance memory, not raw patch or final-layer features |
| Projection provenance | Learned for the heatmap task | Inherited from the earlier synthetic-only instance/pose training and frozen | Task/domain bias is plausible; information loss has not been causally established |
| Context | Three synchronized views jointly encoded | Actual single-view re-encoding followed by a frozen cache | View context changed together with the head and feature interface |
| Spatial head | Width 256; two residual blocks containing four 3x3 convolutions; about 3.44M trainable parameters including auxiliary branches | Width 64 shallow neck plus small shared/per-keypoint readout; 102,807/103,687 trainable parameters | The small-head comparison did not test the July spatial architecture |
| Heatmap/decode | Stride 4; upsample logits; global soft-argmax | Stride 4; upsample conditioned features before readout; peak-centered 5x5 local expectation | Same output stride does not imply the same readout or decoding behavior |
| Instance localization | One head-fixed animal; no detector/query/box conditioning | Six instance queries; predicted boxes/tokens condition the head; soft outside-box penalty | Detection and semantic localization are separate possible failure stages; the penalty is not a hard crop |
| Supervision | Synthetic labels plus DLC/Anipose real references; likelihood threshold 0.7; heatmap, visibility and 3D-related objectives | Reliable real nose/four-paw labels plus broader synthetic semantics; unknowns masked; 2D Gaussian CE | Label support, source, filtering and losses differ |
| Real-data split | 16 training trials and 6 validation trials; validation animal identifiers also occur in training | FIT/CAL/DEV roles; FR3 CAL repeatedly developed on, with upstream detector FR3/ClimbWall exposure | July is trial-held, not animal-held; October is not clean end-to-end sealed-test evidence |
| Selection and units | Validation-selected checkpoint; 624x416 input pixels | Seeded fixed-endpoint contrasts plus development decisions; long-side-768 input coordinates | Cross-run pixel means are not matched treatment effects |

July head training used 651 sampled synchronized real training examples and 125
validation examples, plus 919 synthetic training scenes and 16 validation
scenes. The synthetic split selected the last scenes after discovery/filtering;
it was not an independently established animal/acquisition split. October FIT
contained 874 real views from 134 groups plus 118 synthetic views, not 992
independent acquisitions. The real five-point and synthetic larger-keypoint
tasks must not be silently treated as identical supervision.

The July evaluation reports finger mean 6.07 pixels on 3,678 retained points,
palm mean 4.72 and head mean 4.37, against filtered DLC/Anipose references. The
presentation's overall 4.95 comes from a different training-validation sample
budget, not the same aggregate evaluation. The checkpoint was selected using
the real predicted-rig 3D validation metric. These are meaningful engineering
results, but not full-denominator, independent human-GT deployment accuracy.
No comparable July P95 or anatomical-swap endpoint was reported there.

### What the October evidence establishes

The initial mixed-training CAL bottom-view tail audit is consistent across all
three seeds:

| Seed | Paw errors over 100 input pixels | Of those, within 32 pixels of the contralateral same-type paw | Detection gate failed | Target outside soft-prior support |
| --- | ---: | ---: | ---: | ---: |
| 17 | 34 | 34 | 14 | 1 |
| 29 | 33 | 32 | 14 | 1 |
| 43 | 31 | 30 | 12 | 0 |

Each seed's bottom slice contains 87 known paw points. Diagnostic categories
overlap. Near-contralateral predictions are an identity-confusion proxy, not a
verified reciprocal-swap rate. Even detection-success bottom points have large
errors; box failure alone cannot explain the tail. Inspection of the exported
bottom-view panels also shows detection-success cases with misplaced semantic
points, alongside better cases. Selected panels are not population estimates.

For the later fixed-real-exposure baseline with synthetic weight 0.25 and sigma
12, three-seed mean conditional real-five-point error is 9.68 pixels on FIT
versus 23.18 on CAL. Bottom mean is 16.02 versus 51.95; the mean of per-seed
bottom P95s is 92.73 versus 227.09. These historical coordinate-error summaries
are not the new full-denominator nearest-rank endpoint with failed predictions
assigned infinity. They show both a residual FIT tail and an acquisition/domain
generalization gap; they do not separate architecture from data causally.

The scale inventory finds median real-bottom body equivalent side about 486
pixels versus 187 in synthetic-bottom views. Synthetic motion sources are
concentrated. There were synthetic bottom views, so absence of all bottom-view
training is not the explanation. Detection adaptation improved joint endpoints
without consistently improving conditional bottom-coordinate error. Narrower
sigma and the small per-keypoint head already have negative controlled results;
corrected-label October 12 repeats did not establish a per-keypoint advantage.
Do not present those interventions, or frozen multiview context already tested
elsewhere, as untested fixes.

October 6 used annotations preceding the October 11 correction. Preserve its
scores as historical evidence; use the corrected manifest for a new comparison.
Label revision alone is not an established explanation for all failures.
Native-memory cache parity was checked, so caching itself is not evidence that
the representation was corrupted. The most supported diagnosis is anatomical
identity confusion plus weak generalization, with an additional detection
bottleneck. A restricted feature interface and spatial readout remain testable
mechanistic hypotheses, not a proven limit of the backbone.

### Minimum validation bridge

**Status (2026-10-17):** H0/H1/H2 completed under the old three-fold
acquisition-held protocol, not the 7-fold LOSO; H3 not run. See
[heatmap_bridge_cycle_20261017.md](heatmap_bridge_cycle_20261017.md). Do not
rerun H0–H2 as described below. For future head comparisons, keep the full
known-label denominator as a reported outcome but do not use a P95 gate that
every arm fails because of a shared detector rejection; compare heads under a
common detector policy and report detection failures separately. The original
proposal text is retained below.

First use existing admitted exported predictions for an identity/localization
decomposition: fixed anatomical identities versus a GT-assisted assignment
diagnostic restricted to mutually known paw labels, stratified by detection
success/failure and acquisition. Preserve overlap and unknown-label cases.
GT-assisted reassignment is a diagnostic lower bound, never an automatic score.
GT-position feature probes already supply the location and therefore do not
measure spatial search, proposal robustness or final identity assignment.
Do not open the old mixed-DEV tensor payload to perform this analysis.

If training is subsequently authorized, use the following three arms. Their H
prefix avoids confusion with the archived A/B/C candidate and DDMS arms.

| Arm | Frozen feature input and trainable readout | Question |
| --- | --- | --- |
| H0 | Existing 256-channel memory plus October-style small heatmap head | Rebuild the baseline on the common corrected protocol |
| H1 | Identical memory plus a July-style spatial residual-convolution head | Does a richer spatial readout recover useful information already available? |
| H2 | Raw final aggregation-layer 2048-channel features from the same frozen encoder, plus the same spatial-head family | Does bypassing the old task-specific compression interface add benefit? |
| H3 | July-recipe transplant: synchronized views jointly encoded, final-layer features, July head, July heatmap/3D-consistency losses | Does the complete recipe that worked in July work on free-behavior data? System comparison, not a single-factor ablation |

Run all seeds 17, 29 and 43 on the LOSO benchmark. H0–H2 are a focused bridge
(nine endpoints per fold); H3 is a system arm whose differences from H0 are
listed, not attributed to one factor. If H3 is clearly better, ablate it
downward (remove joint encoding, remove 3D loss) rather than adding factors to
H0 one at a time. All four arms fit the local GPU and should run before the
H100 window. The original first-split description follows: nine endpoints on
the first registered development split. This is a focused bridge, not a full factorial or a reproduction of all
July conditions. H0/H1 tests the spatial-head design as a whole, not parameter
count alone. H1/H2 tests the feature interface including its trainable input
projection; disclose parameter/FLOP differences rather than claiming a pure
dimension ablation. Use final-layer selection from this architectural question,
not a held-out best-layer search.

Hold the encoder checkpoint, single-view input, image transform, automatic
instance boxes/queries, available information, label support, heatmap loss,
decoder, sampling schedule and fixed training endpoint constant. Keep paired
initial tensors for homologous modules and record unavoidable architectural
differences. Do not transfer the July geometry-adapted checkpoint into H2 alone.
Begin with admitted corrected real five-point data only; defer synthetic mixing,
additional view context, 3D losses and LoRA so they do not confound this bridge.
Keep all synchronized/neighboring frames from a capture in one role. Audit
upstream checkpoint exposure: a held-out head-training acquisition with exposed
upstream modules is conditional evaluation, not end-to-end independence.

Register endpoint cost and continuation/harm thresholds before execution. Freeze
every registered arm's predictions for all three seeds before scoring coverage32,
full-denominator nearest-rank P95, anatomical-confusion proxies and per-paw/camera/acquisition
results. Count missing/rejected/nonfinite predictions as misses and infinite
error. Report 704x768 reference-coordinate and original-image errors separately.
Keep detected-only metrics and GT-assisted diagnostics separate. Retain the
90% coverage32 / P95 at most 32 reference pixels deployment target without
equating it with a research continuation gate.

| Observed result | Next justified direction |
| --- | --- |
| H1 improves over H0; H2 adds little | Confirm spatial-readout improvement on further acquisition holdouts |
| H2 improves further | Investigate the old compression/interface, with capacity accounting, before encoder adaptation |
| FIT improves but held-out acquisitions do not | Prioritize acquisition/appearance/pose coverage and independent data admission |
| Conditional pose is good but full-denominator results remain poor | Isolate automatic instance localization and detection failures |
| All arms retain anatomical confusion | Reassess identity supervision and information ambiguity; then consider a controlled cross-view mechanism |

No branch follows solely from a better GT-position descriptor score. A native
DINO baseline would require its own registered checkpoint, exposure, resolution,
head and budget controls; the July result does not establish such a comparison.
It is not part of this minimum bridge or permission to use collaborator-native
DINOv3.

## Data design and evaluation independence

### Data roles

Use three explicit roles: training, development and sealed testing. New training
and any task-specific pretraining use training data only. Use development data
for architecture selection and registered training decisions. Freeze the final
models, thresholds and selection policy before producing sealed-test scores.

Include head-fixed and freely moving mice as named domains in the inventory.
The first heatmap bridge uses the admitted corrected real five-point cohort;
it does not silently pool July finger-joint labels with free-behavior paw labels.
A later joint-domain experiment requires explicit semantic mapping and separate
evaluation on unseen animals or acquisitions in each domain. Cross-domain
transfer in both directions is conditional on sufficient independent data and
budget. Do not silently treat head fixation as a training label, a fixed
anatomical pose or a reason to exclude difficult paws.

| Evaluation slice | Intended question |
| --- | --- |
| Natural held-out distribution | Does the method improve routine use? |
| New animals and acquisitions | Does the benefit survive subject and session changes? |
| Head-fixed and freely moving, separately | Does pooled improvement conceal harm to a domain? |
| New camera arrangements or rigs, where available | Is the method robust to camera configuration changes? |
| Occlusion, contact, crossing paws and unusual appearances | Does the method reduce semantic and severe tail errors? |
| Missing or degraded cameras and calibration perturbations | How does performance fail when source evidence deteriorates? |

Keep naturally sampled evaluation separate from deliberately enriched stress
tests. Record sampling weights and denominators; do not combine them into an
unexplained aggregate score. Define corruption levels on development data and
freeze them before sealed testing.

### Splits and exposure

- Split by animal where identities are available and by complete acquisition
  session. If animal identities are unavailable, disclose that limitation and
  claim acquisition-level independence only.
- Keep synchronized views, near-duplicate frames and neighboring frames from an
  acquisition within the same partition. The model still receives only a
  single instant of multiview input; grouping neighboring frames prevents
  leakage. Temporal modeling is now permitted as the P3 contrast and must
  respect the same grouping.
- Prefer more independent acquisitions over many correlated frames from one
  recording. Set final dataset sizes after the inventory and annotation audit;
  do not promise a particular count before checking availability.
- Record exposure through the detector, VGGT checkpoint, any task-specific
  encoder adaptation, synthetic pretraining and earlier model selection.
  A new downstream split does not erase upstream exposure.
- Keep old FR3 CAL as development evidence. Register the roles and exposure of
  newly admitted datasets explicitly. Never load the old mixed-DEV `rows.pt`
  cache as a convenient expanded training source.
- If upstream modules are retrained, fit them within the appropriate training
  split. If they remain frozen but have seen a held-out acquisition, label that
  evaluation as conditional on exposed upstream features, not independent
  end-to-end testing.
- Keep outer test results sealed during development. Multiple development
  folds and random seeds do not themselves establish a clean outer test.

### Labels and external data

Preserve the five-point primary task. Map keypoint semantics, units, camera
conventions, image transformations and visibility fields before admitting a new
dataset. Unknown coordinates are not negative labels or verified occlusion.
Store corrected or derived annotations as new versions without overwriting raw
labels or historical splits.

Expand training diversity as well as test diversity if suitable training data
are available. More test data improves measurement; it does not teach the model
new appearances. For learning curves, distinguish more frames from more
independent acquisitions, and report sample exposure as well as update counts.

DANNCE provides mouse data and Rat7M is a possible external resource, subject to
availability, label mapping and use terms. Rat7M contains rats; evaluate
cross-species transfer separately rather than pooling it with mouse test
scores. No public dataset is assumed to match all five landmarks or to provide
a head-fixed test domain.

Use reliable 3D annotations where available. Manual 2D annotations triangulated
with the same camera calibration provide a useful reference with uncertainty;
they are not automatically independent physical 3D ground truth. Report
reprojection accuracy and trusted 3D accuracy as different outcomes. Low-quality
surface point clouds must not become strong landmark labels.

## Track A Candidate identity correspondence

Conditional follow-up after the heatmap bridge or an independently supported
candidate-selection hypothesis. This is no longer the mandatory first track.

### Core comparison

Freeze the candidate pool, candidate extraction, calibration, image coordinate
transforms and VGGT backbone for the first comparison. Establish pool recall on
the expanded training and development distributions before attributing errors
to ranking.

| Arm | Input and intervention | Role |
| --- | --- | --- |
| A0 | Candidate confidence and soft geometry | Matched control |
| A1 | A0 plus learned correspondence from local candidate appearance across cameras | Core identity test |
| A2 | A1 plus body appearance context | Conditional, one additional information source |
| A3 | A1 plus candidate-specific source-view reliability | Conditional, separate reliability test |

Define the capacity and compute matching of A0/A1 explicitly. Where possible,
use the same scoring-module shape and paired initial tensors, with appearance
inputs masked for the control. Report unavoidable parameter or compute
differences. Keep sampling, optimizer, endpoint and available labels matched.

Learn same-paw correspondence using training annotations. Include an unmatched
state. Use contralateral and fore/hind confusions as difficult negatives only
where identity is known. Local identity matching, additional context and
difficult-negative sampling are distinct factors: hold the negative sampler
fixed for the core comparison, then change it in a separate registered ablation
if needed. Do not introduce all three factors in one arm.

SuperGlue is a mechanism reference for context aggregation and partial feature
matching. Its pretrained general-image weights are not a validated mouse-paw
matcher. Do not enforce spatial exclusion between different paws that can
legitimately overlap, fixed image-left anatomy, hard bone lengths or nearest-2D
distance assignment.

### Candidate-specific source reliability

A3 asks whether a source camera is useful for a particular paw candidate, rather
than estimating one quality value for the whole view. Training targets may
describe candidate correctness or source contribution using training labels,
including cases in which both branches are good or both are bad. Unknown labels
must not become negative examples. At inference, ground-truth availability,
coordinates, identity assignments and utility values are unavailable.

Measure final selections, rescued errors, newly introduced errors and P95/P99.
Pairwise ranking metrics are diagnostics. Test correct versus shuffled source
evidence and a self-only control. Predefine the shuffle procedure so that it
tests source use without leaking test annotations.

## Track B Learned 3D feature fusion

Conditional follow-up. A positive GT-position feature diagnostic alone does not
justify launching this track ahead of the direct spatial-readout comparison.

Retain VGGT and project multiview image features into a common calibrated 3D
representation. Start with local sparse 3D queries or a coarse volume. Predict
five semantic landmark distributions, allowing uncertainty or multiple
hypotheses, rather than assuming every sharp peak is correct.

| Arm | Intervention | Role |
| --- | --- | --- |
| B0 | Fixed aggregation of the registered projected features | Matched geometric aggregation control |
| B1 | Learned aggregation over the same projected features and spatial support | Core 3D fusion test |

Keep the crop/localization mechanism, feature sampling, spatial support,
calibration, output semantics and supervision matched within B0/B1. A volume
center or proposal must come from inference-available inputs, not ground-truth
landmark coordinates or animal count. Audit pose and coordinate transforms
before training. Both arms train the same downstream readout from paired
initial states; the aggregation mechanism is the registered intervention.
A learned shared 3D representation is a different architectural
family from Track A; comparisons between the families are system comparisons,
not a claim that only one factor changed.

DANNCE and VoxelPose motivate learned reasoning after geometric projection.
Use them as references; any adapted implementation is a separate baseline with
its own label requirements, exposure and compute record. Do not promise a full
paper reproduction within this allocation.

Preserve self-only and shuffled-source checks. Separate a pure-source protocol,
which excludes target images, masks and labels from predictor inputs, from a
target-view refinement protocol. Name the information each arm receives. Loss
labels may be used only within the training partition.

Start with a fixed, documented supervision recipe. Add supervised reprojection
as a separate ablation if warranted; do not change the representation, losses
and augmentation together. Independent trusted 3D evaluation is required for a
physical 3D accuracy claim.

## DDMS inspired representation learning

Historical mechanism proposal, retained with its controls. The local precursor
has since completed descriptor and automatic-endpoint experiments. Its handoff
records that GT diagnostic gains did not rescue the negative automatic result.
Read those frozen outputs before proposing any repeat. This section is no
longer the next execution step; it does not override the revised P0 validation
priority or authorize another sweep.

[DDMS v2](https://arxiv.org/html/2608.23850v2) combines 2D foundation features
with frozen Depth Anything 3 geometry features, learns discriminative
correspondences with a semantic anchor, then distills the teacher into a
single-view student. Its Table 4 shows why ranking alone is insufficient:
direct ranking fine-tuning increases ScanNet correspondence PCK at 10 pixels
from 21.72 to 67.14 while reducing SPair semantic PCK from 68.56 to 57.36.
The complete method reports 66.01 and 75.46 respectively. These are different
benchmark coordinate systems and are not MouseGeoForge endpoint metrics.
The paper selects an intermediate DA3 layer; its layer number does not transfer
to VGGT. As checked on 2026-10-16, the [official repository](https://github.com/ubc-vision/ddms)
and [project](https://ubc-vision.github.io/ddms/) list code and models as coming
soon. A local adaptation is a mechanism test, not a faithful reproduction.

### Proposed MouseGeoForge controls

First compare the existing single-view patch encoder, the four already exposed
VGGT aggregation layers (zero-based 4, 11, 17 and 23), and the current 256-channel
pose memory. Use exactly the same registered same-instant paw correspondences,
contralateral negatives, preprocessing and sampling. Report both native feature
dimensions and a common fixed projected dimension with seeds 17, 29 and 43.
Preserve all layers in the result, rather than reporting only the best layer.
Select any subsequent layer using training-side diagnostics only; disclose
selection when interpreting later development results. GT-position descriptor
probes are conditional diagnostics, not automatic localization measurements.

For any later, separately justified representation experiment, the original
control definitions were:

| Arm | Intervention | Comparison purpose |
| --- | --- | --- |
| C0 | Frozen encoder with the common candidate matcher | Reference |
| C1 | Limited image-encoder adaptation with correspondence ranking | Representation adaptation |
| C2 | C1 plus an anchor to the frozen original feature space | Semantic retention |
| C3 | C2 plus frozen VGGT multiview context in a residual fusion module | Geometric context |

Use the same candidate pool, calibration, crop mechanism, label support,
negative sampler, image schedule, matcher, endpoint and paired initialization.
C1/C2 was the original minimum treatment/control comparison. Reserve an identical
context module and initialization for C2/C3 with context inputs disabled in C2;
record actual trainable counts and FLOPs. C0 intentionally freezes adaptation
parameters, so it is a system reference rather than a claim of identical
trainable capacity. Do not change hard-negative sampling in the anchor contrast.

Use annotated same-paw identity across synchronized views for sparse supervision.
Different known paws can touch: spatial proximity alone cannot make them
positives. Unknown identities are neither negatives nor absence labels. Dense
surface correspondence requires audited synthetic or measured geometry and
visibility; noisy VGGT depth is not a strong label. Do not copy the paper's
centimeter thresholds to mouse anatomy. Keep anatomical identity and physical
surface correspondence as distinct supervision types.

Image-encoder adaptation must read and encode pixels and update the intended
encoder parameters. Verify nonzero finite gradients, changed adapter weights,
frozen backbone checksums, and pixel/feature provenance. A learned projection
on cached 256-dimensional memory does not satisfy this contrast. Frozen
geometry context can be cached only with its exact encoder/preprocessing hash.

Judge automatic candidate selections and full-known localization, including
identity confusions, tail errors and per-acquisition harm. Descriptor ranking
and similarity margins only explain outcomes. Require self-only and shuffled
source controls for C3. Keep target-view refinement separate from pure-source
prediction. If a teacher does not improve the actual endpoint, do not spend the
remaining allocation on student distillation. A single-view student is a later
cost/missing-camera experiment, not an assumed cure for visual ambiguity.

This route originally addressed changes absent from the earlier frozen-memory
metric and peer-view utility-ranking experiments. The completed local evidence
must now inform any successor. Do not reopen failed variants or add another
unrestricted loss scan. Any justified C1/C2 or C2/C3 successor replaces a
lower-priority complete comparison within the measured budget; it must not be
silently added to the nine-endpoint heatmap bridge. Prioritize confirmation.

## Completed 16 hour local precursor

The local run was `ddms_identity_cycle_20261016`. Its registered start was
2026-10-16 21:08:51 UTC, optimization stop 2026-10-17 12:38:51 UTC, and hard stop
2026-10-17 13:08:51 UTC. The local handoff at 12:41 UTC records
`closed_final_report_frozen_automation_paused`; the supervisor records
`stopped`. The closeout records owned-runtime shutdown, GPU/shared-lock
inspection, report freeze and automation pause. Do not resume this window or
append to its queue. Subsequent local work uses a separate bounded namespace.

Authoritative local records are `output/ddms_identity_cycle_20261016/plan.json`,
`next_actions.json`, `heartbeat_launch_handoff.json`, `state.json`,
`supervisor.json`, and `final_closeout_v12/report.json`. Preserve their history.
The observed workstation has one NVIDIA RTX PRO 6000 Blackwell Max-Q GPU,
reported by the driver as 97,887 MiB. This local allocation is separate from
the planned four-or-more H100 allocation.

The handoff records completed provenance/layer diagnostics, actual pixel
re-encoding and encoder-gradient feasibility, paired encoder experiments,
negative-type, source-quality, coordinate-jitter and proposal-objective
experiments, independent numerical audits and actual image review. Technical job
counts do not count independent experiments. Preserve every seed, failure and
partial result. A better GT-coordinate representation diagnostic did not by
itself establish better automatic keypoints. Use the frozen endpoint reports,
not descriptor metrics, for any further continuation decision. The original
five-point selector remains an archived reference.

The registered corrected real FIT subset contained 134 synchronized groups,
874 views and 3,085 known five-point annotations (2,432 paw annotations).
FR3 CAL remains repeated development; it was not used in the initial FIT probes.
The user's broader-data permission supports metadata inventory and subsequent
explicit admission manifests; it does not make old DEV-mixed caches a valid
source. Before loading a new payload, fix its training/development/sealed-test
role, semantic mapping, acquisition split and upstream exposure. As of
2026-10-17 the user permits GR2, UR2 and UR3; admit them explicitly as the P0
sealed test, never silently and never as training data. Historical DEV
payloads remain unadmitted.

The closed run used its own supervisor/queue and the shared local GPU lock at
`output/teacher_temporal_cycle_20261001/gpu_job.lock`. Keep its STOP files, frozen
scripts, protocols, checkpoints and raw labels intact. Do not restart the
paused `pymousestudio` automation as part of this document update. Any future
authorized run needs a new namespace and deadline; verify boot ID, start ticks
and process group for process control. No model promotion or real pseudolabel
publication is authorized by this plan; internal training use of
multiview-consistent pseudo-labels is permitted under P2.

## Conditional encoder and data experiments

Only after a core architecture shows useful development behavior, compare the
following four cells on that architecture:

| Encoder regime | Original-scale training subset | Expanded diverse training set |
| --- | --- | --- |
| Frozen VGGT encoder | Reference | Data effect |
| Adapted VGGT encoder | Adaptation effect | Interaction between data and adaptation |

Use adapter/LoRA or selected final layers first. Choose one adaptation method
before launching the comparison; full encoder fine-tuning requires adequate
data and a favorable measured resource profile. Preserve the VGGT backbone and
the existing exclusion of collaborator-native DINOv3.

The smaller training subset must be a registered subset of the larger training
partition, not a different test cohort. For the primary comparison, fix the
compute budget and report how image exposure changes. If a convergence-matched
comparison is also run, report it separately. Equal optimizer steps do not imply
equal data exposure or equal convergence.

Cached features are valid for a frozen encoder. Encoder adaptation and pixel
masking require image re-encoding; verify a nonzero intended gradient and actual
pixel changes. Masking a cached feature tensor is not pixel occlusion.

LP3D-inspired fusion and pixel masking may be revisited only with a new,
registered hypothesis about data diversity or encoder adaptation. Keep the
fusion-by-pixel-masking 2x2 comparison intact if it is run. Temporal models and
EKS are now in scope only as the P3 controlled contrast.

Targeted synthetic training is also conditional. First verify the pipeline can
produce trustworthy identity, correspondence and occlusion labels. Compare
targeted versus random synthetic data with equal real-data exposure; keep an
extra-real control separate and report its different exposure. More renders
from the same few scenes do not establish greater diversity.

## Compute allocation and run matrix

The following is a future scheduling envelope, not execution permission or a
measured training-time estimate. It replaces the earlier mandatory A/B matrix.
Set absolute UTC deadlines when the run starts; the hard stop is T0 plus 48
hours, with optimization stopping at T0 plus 40 hours. Additional GPUs increase
parallel capacity, not the authorized wall-clock duration.

P0 and P1 (sealed selector test, LOSO harness, LP baseline, chirality-aware
identity, H0–H3) run on the local GPU before the H100 window; their results
choose the P2 starting system. The H100 window is then:

| Elapsed time | Priority | Required output |
| --- | --- | --- |
| 0-4 hours | Data/exposure audit of admitted training sources, P0/P1 system replay on the nodes, loading and multi-GPU profiling | Frozen manifests, verified replay, measured throughput |
| 4-12 hours | P2 pseudo-label generation over unlabeled synchronized video with the P0/P1 system; consistency filtering | Versioned internal pseudo-label set with acceptance statistics per acquisition |
| 12-34 hours | P2 training: multiview VGGT adaptation (LoRA/partial) on real + pseudo-labeled + admitted external + synthetic data, with a matched no-pseudo-label control; all three seeds | Registered endpoints scored on LOSO folds |
| 34-40 hours | P3 temporal on/off contrast on the best P2 system | Identity-swap and >100 px tail comparison on LOSO |
| 40-48 hours | No further optimization; frozen inference and audit | One sealed scoring of the final frozen policy, visual review, resource accounting and final report |

Use seeds 17, 29 and 43. Measure cost before fixing each endpoint. If P2 cannot
fit, shrink the data volume of both arms equally before launch; never drop an
unfavorable seed or complete only a treatment. Track A/B and DDMS successors
remain conditional and must not displace P2/P3 or closure.

The GR2/UR2/UR3 sealed test is scored once for the frozen P0 selector before
the H100 window. Any H100-era final policy needs its own sealed evaluation;
use a newly admitted held-out acquisition if available, otherwise report the
GR2/UR2/UR3 rescore as test evidence reused after a single prior look.

Parallelize independent seeds, folds and matched arms first. Use within-node
multi-GPU training when a model requires it or a benchmark demonstrates a net
speedup. Use cross-node synchronous training only after checking the actual
interconnect and data pipeline. Eighty GB does not guarantee that an arbitrary
full VGGT multiview training configuration fits.

Record GPU-hours, wall time, peak memory, images/second, camera views per example,
optimizer updates, unique acquisitions and repeated image exposure. Reduce
scope early if measured completion estimates threaten the confirmation and
audit budget. Any shortened endpoint must be fixed for both arms before launch;
do not retroactively compare a partial run with a completed control.

## Metrics and decision rules

### Primary reporting

Report nose and four paws jointly and separately, by anatomical point, camera,
animal/acquisition and head-fixed/free-moving domain. Include worst-group
results with their sample counts, identity confusion proxies, overlap cases and
large-error tails.

- Coverage within 8, 16 and 32 pixels, with the full known-label denominator.
- Median, nearest-rank P95 and P99 error; report missing/rejected/nonfinite
  predictions explicitly and count them as misses. For full-denominator
  quantiles, treat failed known-point predictions as infinite error.
- Input-coordinate pixels at the registered 704x768 reference transform and
  original-image pixels separately. For new aspect ratios or preprocessing,
  freeze and document the transform; do not silently reuse the same numerical
  threshold across incomparable scales.
- Millimeter errors only on datasets with an audited physical scale and
  appropriate 3D reference, including the reference's uncertainty.
- Candidate recall, descriptor/ranking scores, reprojection consistency and
  source-use checks as explanatory diagnostics, not substitute endpoints.

Use paired uncertainty summaries at the independent animal or acquisition level,
according to the split. Do not count seeds or correlated frames as independent
animals. Report all registered seeds and failures, and account for development
selection and multiple comparisons when interpreting intervals.

### Continuation and acceptance

Separate a research continuation decision from a deployment-quality decision.
Before core runs begin, register the minimum useful endpoint improvement,
allowable per-domain harm, tail-error condition and source-use check. Select
these criteria using the intended task and development protocol, never by
looking at sealed-test scores.

A method may justify larger-data research if it repeatedly improves final
identity/localization and tail behavior without unacceptable domain harm, even
while its absolute quality remains inadequate. That does not change the failed
status of previous variants or make a new model ready for deployment.

Retain the previous final quality target of at least 90% coverage within 32
reference-input pixels and nearest-rank P95 at most 32 pixels, and report domain
and difficult-slice results alongside it. Freeze any additional acceptance
requirements before the run. Unknown labels are excluded from the known-label
denominator; failures on known labels are not excluded.

All final model and normalization states must be frozen before sealed-test
inference, and all registered final prediction outputs must be frozen before
scoring. Do not select seeds, epochs, thresholds or ensemble membership using
test results. Do not promote a checkpoint or publish real pseudolabels
automatically.

## Execution records and recovery

Create a new run namespace, for example
`output/h100_multiview_validation_<start_date>/`. Do not reuse the old
single-GPU supervisor or presume its local lock coordinates a cluster. Use the
cluster scheduler's GPU allocations and a new run-owned queue.

Each job record must identify:

- Run ID, arm, seed, fold, dependency IDs and registered endpoint.
- Git commit plus any source-diff hash, environment/container version and
  immutable protocol hash.
- Dataset/split manifests, label definitions, calibration/transforms, input
  hashes and upstream exposure record.
- Initial weights, paired initialization, trainable modules and optimizer
  configuration; record trainable and total parameter counts.
- Scheduler job ID, host/node identity and allocated GPU identities. For local
  process control, also verify boot ID, process start ticks and process group.
- Start/end timestamps, deadline, resource usage, sample exposure, checkpoint
  hashes, prediction hashes, failure status and audit outputs.

Freeze caches by split, encoder checksum, preprocessing and data-manifest hash.
Never reuse a frozen-feature cache for an adapted encoder. Keep test labels out
of predictor inputs and training caches.

Preserve failed runs. Use new script/protocol versions and job IDs for repairs;
do not rewrite a frozen experiment. On restart, reconcile scheduler state and
process ownership before launching replacements. Stop only jobs owned by the
new run. Close the run within its hard deadline without relying on a later
heartbeat, and verify owned workers and GPU allocations have been released.
CPU work obeys the same deadline.

The continuing exclusions are internal spine targets, collaborator-native
DINOv3, hard bone-length constraints and fixed image-left identity rules
(body-frame 3D handedness under P1 is not an image-left rule). Temporal
modeling and EKS were removed from the exclusions on 2026-10-17 and are
permitted as the P3 contrast. Preserve raw labels and old checkpoints. Keep
data, weights, rendered assets and experiment outputs out of Git; publish only
authorized source and documentation.

## Required handoff and final deliverables

The receiving agent should produce a protocol and data-readiness report before
long training, with unresolved allocation or dataset issues stated explicitly.
At completion, provide:

1. Dataset inventory and split/exposure manifests with counts of animals,
   acquisitions, synchronized groups, views and known points for each domain.
2. Exact experiment matrix, all registered outcomes, failures and deferred work.
3. Baseline and paired endpoint metrics, per-domain results, uncertainty
   summaries and resource/exposure accounting.
4. Fixed-rule visual examples of successes, regressions, contralateral/fore-hind
   swaps, overlap, missing views and tail errors. Record which images were
   actually inspected; generating a contact sheet is not an inspection receipt.
5. Frozen configurations, source/checkpoint/output hashes and replay results.
6. A final decision explaining whether the spatial readout or feature interface
   improved unseen-acquisition behavior, and whether the remaining bottleneck
   is identity, instance localization or generalization. Report conditional
   identity/fusion/data/encoder experiments separately if conducted. Distinguish
   development findings from sealed-test evidence.
7. A closeout record confirming the optimization stop, owned-job shutdown and
   final report freeze.

The successful outcome is a reproducible explanation of which intervention
improves generalization and severe-error behavior. A larger job count alone is
not a scientific result.

## Local evidence entry points

The following paths are relative to the original repository root. They are
local-only evidence references, not files promised to exist in a GitHub clone.
Read summaries and manifests first; do not open unregistered payloads merely
because a historical script mentions them.

| Local path | Purpose |
| --- | --- |
| `vggt-omega/training/run_keypoint_head_all29_624x416.sh` | July checkpoint, trial split, supervision and optimization settings |
| `vggt-omega/vggt_omega/models/heads/keypoint_head.py` | July final-layer spatial heatmap architecture |
| `output/blenderpymouse_vggt_omega_all/run_20260728_keypoint_head_all29_frozen_624x416_e5/evaluation_best/summary.json` | July filtered-reference evaluation and denominators |
| `output/keypoint_head_cycle_20261006/head_ab_seed17_v1/source/keypoint_adapter_readout.py` | Immutable October head implementation, conditioning and decoder |
| `output/keypoint_head_cycle_20261006/head_ab_seed17_v1/protocol.json` | October paired schedules, supervision and feature contract |
| `tools/prepare_single_view_context_20261004.py` | Actual single-view feature preparation and checkpoint chain |
| `output/surface_curve_cycle_20261003/training_v1/protocol.json` | Lineage to the earlier synthetic-trained instance-head projections |
| `output/pose_diverse_core23_768_v1/manifest.json` | Synthetic-only predecessor training provenance |
| `output/keypoint_head_cycle_20261006/pose_failure_audit_v1/results.json` | Three-seed tail/identity/detection attribution |
| `output/keypoint_head_cycle_20261006/decision_heatmap_fit_gap_results_v1.json` | FIT/CAL gap, historical metric definitions |
| `output/keypoint_head_cycle_20261006/training_view_scale_audit_v1/results.json` | Real/synthetic scale and view support |
| `output/keypoint_head_cycle_20261006/final_dev_visual_review_v1/head_architecture_seed17/examples_02.jpg` | Actually inspected October bottom-view examples |
| `output/ppt_figures_vggt_keypoints_20260728/01_keypoint_prediction_real.png` | Actually inspected July examples; mixed metric populations require the caveat above |
| `output/ddms_identity_cycle_20261016/next_actions.json` | Completed local experiment/audit inventory and no remaining queued plan |
| `output/ddms_identity_cycle_20261016/final_closeout_v12/report.json` | Frozen closeout, ownership/GPU checks and automation pause |
| `output/ddms_identity_cycle_20261016/final_closeout_v12/freeze_receipt.json` | Final report freeze provenance |
| `output/semantic_matching_cycle_20261015/final_closeout_v1.json` | Authoritative lifecycle status, main selector result and limitations |
| `output/semantic_matching_cycle_20261015/final_audit_reports_v37/20261016T073618169304Z/index.html` | Frozen final report |
| `output/semantic_matching_cycle_20261015/final_audit_reports_v37/20261016T073618169304Z/frozen_manifest.json` | Frozen report provenance |
| `output/semantic_matching_cycle_20261015/sparse_fusion_failure_decision_v1.json` | Ranking versus candidate-pool misses and identity proxies |
| `output/semantic_matching_cycle_20261015/shared_3d_compatibility_decision_v1.json` | Common 3D candidate feasibility bound and its limitations |
| `output/semantic_matching_cycle_20261015/native_early_fusion_2x2_decision_v1.json` | Early fusion and actual pixel-occlusion outcomes |
| `output/semantic_matching_cycle_20261015/peer_view_ranking_decision_v1.json` | Ranking-loss endpoint comparison |
| `output/semantic_matching_cycle_20261015/paw_metric_transfer_decision_v1.json` | Targeted/random synthetic and extra-real descriptor comparison |
| `output/multiview_semantic_matching_review_20261015/review_v1.json` | Registered literature review and mechanism controls |

## Primary method references

- [SuperGlue official implementation](https://github.com/magicleap/SuperGluePretrainedNetwork):
  context aggregation and partial matching of sparse image features.
- [DANNCE original paper](https://www.nature.com/articles/s41592-021-01106-6):
  geometric projection followed by learned 3D reasoning for animal landmarks;
  includes mouse data and Rat7M availability information.
- [DANNCE official repository](https://github.com/spoonsso/dannce):
  data formats, calibration, labels and reproduction dependencies.
- [VoxelPose official implementation](https://github.com/microsoft/voxelpose-pytorch):
  learned multiview reasoning in a common 3D representation for human pose;
  mouse adaptation is a separate experiment.
- [Lightning Pose documentation](https://lightning-pose.readthedocs.io/en/stable/):
  reference for multiview fusion, pixel-masking and temporal context/EKS;
  P0 external baseline and P3 temporal contrast.
