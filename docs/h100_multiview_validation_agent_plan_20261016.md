# MouseGeoForge H100 multiview validation plan and agent handoff

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

Created: 2026-10-16. Updated: 2026-10-17 after the local cycle closeout and the
July Water Reaching versus October 6 free-behavior implementation audit.
Status: planning and evidence review only. The user's current instruction is
**do not start new training**. The H100 allocation has not started, and the
16-hour local precursor is closed with its automation paused.

The immediate research priority is to test the direct spatial heatmap readout
path before further candidate-ranking or encoder-adaptation work. The July
success and October failure did not use the same encoder checkpoint, feature
interface, spatial head, view context, supervision or evaluation population.
Their difference does not establish a limitation of VGGT or DINO features.
Keep VGGT for the first controlled bridge. Candidate identity correspondence,
learned 3D fusion and encoder/data expansion remain conditional follow-ups.
This revision supersedes the previous order of those tracks; it authorizes no
new run, supervisor, queue, allocation or dataset admission.

The user expects at least four NVIDIA H100 GPUs, provisionally 80 GB each, for
48 continuous hours. Four GPUs provide a nominal ceiling of 192 GPU-hours;
initialization, data loading, evaluation and failures consume part of that
budget. Node topology, interconnect, storage, scheduler and actual availability
are still unspecified. The user permits larger and more diverse datasets,
subject to the admission and exposure requirements below. The October 16
authorization applied to one bounded local cycle, which is now closed; it does
not allocate the H100 cluster or extend either completed local cycle. The
current request updates the route document only.

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
5. Complete the read-only error decomposition on admitted, exported predictions.
   Separate anatomical identity, spatial localization and instance detection;
   keep GT-assisted diagnostics out of the automatic endpoint.
6. If the user later authorizes execution, verify admitted inputs and baseline
   replay, profile a representative single-GPU job, and register the heatmap
   bridge before launching it. Resolve allocation, absolute start/stop times,
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

The next controlled work, if authorized, should answer:

1. Does a stronger spatial heatmap readout improve automatic localization and
   anatomical identity using exactly the existing frozen memory?
2. Does bypassing the old synthetic-task compression interface improve that
   readout further, with the encoder and inference information held fixed?
3. Do any gains survive acquisition holdouts, rather than only improving FIT?

Only then choose a justified follow-up: data coverage, instance localization,
cross-view identity, learned 3D fusion or encoder adaptation. Do not spend the
allocation on a broad loss, temperature, resolution or head-size sweep.

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

### Minimum validation bridge, proposed only

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

Run all seeds 17, 29 and 43: nine endpoints on the first registered development
split. This is a focused bridge, not a full factorial or a reproduction of all
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
  leakage and does not authorize temporal modeling.
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
longer the next execution step; it does not override the no-training instruction
or the heatmap bridge priority.

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
append to its queue. The current instruction explicitly prohibits new training.

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
role, semantic mapping, acquisition split and upstream exposure. Never silently
admit the previously protected GR2, UR2, UR3 or historical DEV payloads.

The closed run used its own supervisor/queue and the shared local GPU lock at
`output/teacher_temporal_cycle_20261001/gpu_job.lock`. Keep its STOP files, frozen
scripts, protocols, checkpoints and raw labels intact. Do not restart the
paused `pymousestudio` automation as part of this document update. Any future
authorized run needs a new namespace and deadline; verify boot ID, start ticks
and process group for process control. No model promotion or real pseudolabel
publication is authorized by this plan.

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
EKS remain outside scope.

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

| Elapsed time | Priority | Required output |
| --- | --- | --- |
| 0-6 hours | Data/exposure audit, baseline replay, loading and GPU profiling | Frozen manifests, verified baseline, feasible endpoint budgets |
| 6-24 hours | H0/H1/H2 heatmap bridge, all three seeds | Registered fixed endpoints and full-denominator development evaluation |
| 24-40 hours | Confirmation on additional held-out acquisitions/folds | Repeated results for the most promising complete comparison; conditional additions only if budget remains |
| 40-48 hours | No further optimization; frozen inference and audit | Sealed-test results, visual review, resource accounting and final report |

Use seeds 17, 29 and 43. H0/H1/H2 require nine endpoints on one registered
development split. Measure cost before fixing the endpoint. If the bridge cannot
fit, reduce its common endpoint before launch or defer a complete comparison:
H0/H1 across all seeds tests the head first, with H2 explicitly deferred. Never
drop an unfavorable seed or complete only a treatment. This is not a commitment
to exhaust the allocation; preserve confirmation and final-audit time.

For confirmation, prioritize one complete treatment/control pair on up to two
additional acquisition-held-out folds with all three seeds, if feasible. Use
development folds for this selection. Freeze the final model-selection policy
before sealed testing. Tracks A/B, DDMS successors, encoder adaptation, synthetic
training and external baseline reproduction must not displace bridge
confirmation or closure. Adding views or July 3D losses is a separate contrast.

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

The continuing exclusions are temporal/adjacent-frame modeling, EKS, internal
spine targets, collaborator-native DINOv3, hard bone-length constraints and
fixed image-left identity rules. Preserve raw labels and old checkpoints. Keep
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
  reference for multiview fusion and pixel-masking mechanisms; temporal
  components remain outside this plan.
