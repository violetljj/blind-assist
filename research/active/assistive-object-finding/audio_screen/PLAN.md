# Alternative thesis screen: environmental sound and functional parts

2026-10-15 / EXPLORE. The user requested another BLV-related thesis direction
while Ego4D approval is pending. This is an independent feasibility screen, not
an extension or retuning of the completed object-finding pilot. Product reading
is owned by a concurrent task and is outside this screen.

## Question, budget and decision

Check whether public real audio, a current pretrained baseline and automatic
evaluation expose a defensible research problem. In parallel, audit functional
part / action-affordance data as a potentially more direct BLV application.
Do not choose the thesis from this screen, claim safe grasping, or claim a BLV
user benefit without appropriate later evidence.

Authorized local bounds: audio/model downloads <=1.65 GB; audio GPU process wall
<=900 s including canary; data/analysis CPU branches <=600 s each. A separate
functional-part data audit may download <=150 MB, CPU <=600 s, until 14:15 HKT.
Finish this screen by 14:30 HKT on 2026-10-15. No paid API, remote allocation,
new hardware or manual annotation. Preserve source receipts, failures, outputs
and exact denominators; release task-owned processes and locks.

## Data and inference fixed before scoring

ESC-50 official commit 33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6. Select 120
different source recordings with stable SHA256(filename + "20261015") order.
Calibration has 60 rows (folds 1/2/3, 20 per fold). Evaluation has 60 rows
(fold 4:40; fold 5:20). Each partition is balanced known/unknown dominant class;
no source crosses partitions. Exact sample ids and original licenses are saved.

Known classes: dog, crying_baby, footsteps, door_wood_knock, clock_alarm,
glass_breaking, car_horn, siren, engine, toilet_flush.

Calibration unknown classes: cat, hen, rain, wind, coughing, laughing,
door_wood_creaks, keyboard_typing, train, vacuum_cleaner.

Evaluation unknown classes: rooster, insects, sea_waves, crackling_fire,
sneezing, clapping, mouse_click, church_bells, helicopter, chainsaw.

**Ground truth supports dominant-label vocabulary rejection only.** ESC-50 is
not exhaustively annotated for all audible events. An unlisted event is not
proven absent; unknown here is not unseen during model pretraining. Do not call
these counts real-world false alarms or query-specific target absence.

CLAP laion/clap-htsat-unfused revision 8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a,
float32, batch 4. Whole 5-second recording and three 2.5-second windows at
0/1.25/2.5 seconds. PCM scaling, scipy 48 kHz resampling and official repeatpad;
no denoising or RMS normalization. Three fixed class prompts are preserved in
class_prompts.json. Inference reads observations only, without truth. Fifty
class scores and a diagnostic background row are saved; open-vocabulary
rejection comparisons below use only the ten declared known classes.

## Scoring specification frozen 14:02 HKT

Inference finished before this document; no prediction values or evaluation
outcomes have been inspected by the primary agent when fixing this analysis.
This is a small Development screen, not a preregistered confirmatory study.

Fit any learned gates on calibration folds 1 and 2 only (40 rows). Calibrate
every gate threshold on fold 3 (20 rows, 10 known / 10 unknown), choosing the
highest threshold retaining at least 9/10 known dominant-label recordings.
Ties are accepted. Evaluate once on untouched folds 4 and 5 (60 rows).

Fixed comparisons:

1. Whole-clip first-prompt ten-class argmax, without rejection.
2. Same predictions, reject using maximum cosine.
3. Same predictions, reject using top-one minus top-two cosine margin.
4. Same three-window budget: classwise maximum first-prompt cosine, argmax and
   maximum-score threshold. This is the required simple local-evidence control.
5. Whole-clip prompt-ensemble argmax with maximum-score threshold.
6. Regularized logistic known-primary-label gate: standardized whole first-prompt
   maximum cosine and margin; C=1, lbfgs, max_iter=1000, random_state=20261015.
7. The same logistic gate plus five fixed observable features: standard deviation
   of the whole-clip winner's three prompt scores, mean and standard deviation
   of that winner's three window scores, fraction of windows agreeing with the
   whole winner, and largest window score minus whole score for that winner.
   Keep the same whole-clip class prediction to isolate gate changes.

Report accepted known /30, correctly classified and accepted known /30,
accepted unknown dominant-label clips /30, and all known top-one accuracy.
Also report known/unknown AUROC and average precision as descriptive diagnostics;
do not pick a new method or threshold on evaluation. Whole 50-class accuracy is
a separate closed-set diagnostic, not a rejection method. State training data
and inference cost differences; no confidence interval implying population
coverage from 30 clips. No synthetic audio mixtures in this run.

If thresholding matches or dominates the richer gate, it is evidence against
selling this gate as a thesis contribution. If temporal evidence adds no gain,
do not rename window pooling as innovation. Natural multi-event presence and
actual BLV utility remain untested regardless of this result.

Payload root: `artifacts.local/work/assistive-audio-pilot-20261015/`.
