# CME minimal core code

Core scripts for the study *A Large-Scale Coronal Mass Ejection Instance
Segmentation Dataset with Catalog-Guided Selection and Two-Stage Human
Refinement*. This version includes LASCO downloads and quality/time selection,
plus runtime-tested fixes for preprocessing, model imports and training state.

## Included scripts

| Script                   | Purpose                                                                         |
| ------------------------ | ------------------------------------------------------------------------------- |
| `download.py`            | Request LASCO FITS with SunPy Fido and a Level(1) archive filter                |
| `fix_processed_files.py` | Strict QUALITY_INDEX >= 3 and absolute first-C2 offset <= 3600 seconds          |
| `preprocess.py`          | Running differences, vertical flip, intensity scaling, filtering and solar mask |
| `train.py`               | Original LoRA training and development checkpoint selection                     |
| `infer.py`               | Initial binary masks using SAM ViT-B and rank-512 LoRA                          |
| `png_to_labelme.py`      | External-contour polygons for initial LabelMe annotations                       |

The required local `src/` implementation is retained. Demos, overlays, cached
files, observations and binary model files are omitted. All Python comments
and docstrings in this package are in English. Provenance and changes are
recorded in `SOURCE_MANIFEST.json`.

## Installation

Use a separate environment and run commands from the package root. This
release was runtime-tested on an existing Python 3.9.16 / PyTorch 2.0.1
environment. Python 3.10/3.11 are also compatible targets for the specified
PyTorch generation, but fresh installations on them were not tested here:

```sh
python -m pip install -r requirements.txt
```

The ranges are compatibility guidance, not a recovered historical lock file.
The supplied project specified torch 2.0.1 and torchvision 0.15.2. For a GPU,
install the matching build using https://pytorch.org/get-started/previous-versions/.
The exact tested versions are recorded in RUNTIME_CHECKS.json. These
checks used an existing environment with a small isolated dependency overlay,
not a clean installation. The selection script uses only the standard library.

## 1. Quality and one-hour event selection

Inputs are local text files containing event blocks: the catalogued first C2
appearance time, its quality index, and height-time measurement rows. Original,
processed and simplified block formats from the supplied project are accepted.
`examples/catalog_example.txt` is a synthetic format example, not real data.
Monthly HTML catalog pages are not accepted directly. The script does not
create these aggregated input files or retrieve event-specific catalog records;
their acquisition/source records must be documented separately.

```sh
python fix_processed_files.py --input ./examples/catalog_example.txt --output ./selected_events.txt
python fix_processed_files.py --input ./records/2002.txt ./records/2003.txt --output ./selected_events.txt
```

Use the first command for the included example, or the second with your own
records. Output consists of selected event blocks and `selected_events.audit.json`
with a decision and rejection reason for every parsed event. Existing outputs
can be replaced; input files cannot be overwritten by the script.

Selection rules, in order:

1. Require an integer QUALITY_INDEX >= 3. Missing, invalid, conflicting, or
   below-threshold values are rejected. Unlike the supplied historical script,
   a missing quality index is not retained.
2. Compare the catalogued first C2 appearance time with the earliest timestamp
   among C2 measurements in that event's supplied record. C3 rows do not
   determine the first C2 time. Input order does not determine chronology.
3. Retain the event when the absolute difference is <= 3600 seconds. Exactly
   one hour is retained; 3600.001 seconds is rejected. Offsets can be positive
   or negative because this is an absolute-difference rule. Timestamps without
   a timezone are UTC; explicit offsets are converted to UTC. Date boundaries
   are handled using full datetimes, not time-of-day subtraction.
4. Reject events without a C2 observation or with an invalid required timestamp.
   An invalid C2 timestamp is not silently skipped.

This is an event-level filter: all measurement rows of a retained event are
preserved, including later C2 observations more than one hour after onset and
any C3 rows. Subsequent acquisition/preprocessing must select C2 observations.
The audit's first time is based on the supplied records, not on inspected FITS
headers or server availability. If the intended comparison is the first
actually retained FITS frame, supply the corresponding retained-C2 records
and recheck the offset after file eligibility checks.

The temporal selection is newly implemented for this release on 2026-10-22;
it was not present in the supplied scripts. Strict handling of missing quality
is also a revision. This release does not assert that these revisions have
reproduced the original study's published event counts.

## 2. LASCO download

```sh
python download.py --start 2023-07-29T12:00:00 --end 2023-07-29T12:30:00 --output ./downloads/event_A --detector c2
```

This example searches LASCO/C2 processing Level 1 within the requested interval
using Fido.search and Fido.fetch. It queries all matching observations in the
interval and does not automatically restrict results to individual catalogued
height-time measurement timestamps. Match files to the selected event's C2
records, check retrieval/resolution requirements, and calibrate any non-Level-1
inputs before preprocessing. A live query downloaded one 1024 x 1024 LASCO C2
FITS file during testing. Its header did not explicitly identify the processing
level; calibration was not independently established. Do not infer physical
calibration solely from the Level(1) query filter.

## 3. Preprocessing and initial annotations

Place selected, calibrated 1024 x 1024 LASCO C2 `.fts` files in separate event
directories with chronologically sortable filenames. Obtain the official
`sam_vit_b_01ec64.pth` checkpoint from
https://github.com/facebookresearch/segment-anything#model-checkpoints and put
it at the package root. Put the separately supplied `lora_rank512.safetensors`
in `lora_weights/`. Both binary files are excluded from this code ZIP; the
same LoRA file supplied with version 1 is used, with SHA-256 in the manifest.

```sh
python preprocess.py --input ./selected_level1 --output ./processed_png
python infer.py --input ./processed_png --output ./predicted_masks
python png_to_labelme.py --masks ./predicted_masks --images ./processed_png --output ./proposals
```

Keep input and output trees separate. The first eligible FITS frame in each
directory supplies the reference; later frames are differenced against the
previous eligible frame. The original preprocessing explicitly skips 512 x 512
images and does not itself implement all resolution/retrieval criteria.
Inference requires 1024 x 1024 images.

Numerical preprocessing is unchanged: vertical flip; log10(diff + 1e-12) + 14;
NaN replacement with 1; clipping to [1,5]; uint8 scaling; 3 x 3 median filter;
and a central mask with radius 2.4 apparent solar radii. Negative differences
can produce an invalid-log warning before NaN replacement. No extra rotation
is applied. Inference uses the full [0,0,1024,1024] box and single-mask output.

Open proposal JSONs in LabelMe. Their relative imagePath refers to the matching
running-difference background, including on Unicode filesystem paths. Preserve
the relative layout when moving files. External contours omit holes; contours
with fewer than three points are discarded. Initial labels are `cme` and
group_id is null. Human refinement and final instance assignment are still
required. Intermediate filenames are not the final published file convention.

## 4. Training

Supply the study's image/mask pairs; the original provided training directories
were empty. Each image needs a same-named mask in a sibling `masks/` directory:

```text
dataset/train/event_A/frame.png
dataset/train/event_A/masks/frame.png
dataset/test/event_B/frame.png
dataset/test/event_B/masks/frame.png
```

Edit `config.yaml`, then run `python train.py` from the package root.
Here `test` denotes the development validation set (70 images in the manuscript),
not the independent 100-frame audit. The manuscript reports 32 training images;
keep the development splits disjoint by event. Use binary 0/255 masks.

Training retains the supplied rank 512, batch size 1, 90 epochs, Adam learning
rate 1e-4, zero weight decay and DiceCELoss settings. The log now correctly
labels accuracy_score as PA (pixel accuracy), replacing the misleading MPA
label without changing its calculation. Development evaluation still thresholds
raw logits at 0.5; inference uses the model's 0.0 threshold. Validation now restores
the model's preceding training/evaluation mode, and each training epoch explicitly
enters training mode. These are documented release fixes; they are not asserted
to be the exact historical implementation used to obtain published results. No original random seed
or exact historical environment is available. This script does not reproduce
the independent validation metrics reported in the manuscript.

train.py reads ./config.yaml from the current directory. The unused
configuration read in src/lora.py has been removed, so importing the model
no longer requires a config file. Keep the supplied src tree: its SAM training forward path differs from stock
SAM, including resized logits returned under low_res_logits.

## Licenses

Third-party notices and licenses are retained in THIRD_PARTY_NOTICES.txt and
LICENSES/. The dataset's CC BY 4.0 designation does not replace upstream
software/model licenses. No explicit license for study-specific contributions
or trained LoRA parameters was present in the supplied source folders; this
local preparation does not assign one.
