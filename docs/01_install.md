# 01 — Install

Tested on Ubuntu 22.04 with NVIDIA L20 GPUs (driver with Vulkan support),
conda and Python 3.10. Needs `curl`, `unzip` and `sha256sum`.

The whole flow, from a fresh clone:

```bash
conda env create -f environment.multiRobots.yml
conda activate multiRobots
pip install -e .
bash tools/install/install_assets.sh --download --install-wheel core   # or: all
python tools/install/check_install.py core                             # exit 0 = ready
```

The sections below explain each step.

## 1. Conda environment + package

```bash
conda env create -f environment.multiRobots.yml
conda activate multiRobots
pip install -e .             # installs the followbench + baselines packages (editable)
```

`environment.multiRobots.yml` pins every pip dependency, including torch
(CUDA 13 wheels), ultralytics, the KPR ReID extras and the TrackVLA / OA-VAT
dependencies. One dependency is installed from GitHub
(`cosine_annealing_warmup`). If `conda env create` stops on a network error,
re-run `conda env update -n multiRobots -f environment.multiRobots.yml`; it
continues with the env that was already created.

The CARLA client is not in the YAML: it must match the simulator build, so it
is installed from the wheel shipped with the build (step 2, `--install-wheel`).

## 2. Download the release assets

The custom CARLA build and the model weights are not in git. They are four zip
files in the Hugging Face dataset
[`hjyeee/Follow-Bench-2.0-packages`](https://huggingface.co/datasets/hjyeee/Follow-Bench-2.0-packages):

| Package | File | Download | Unpacked | Needed for |
|---|---|---|---|---|
| `carla` | `followbench2_carla_multi_robots.zip` | 5.8 GB | 14.2 GB | everything (CARLA 0.9.16 + Scout / Dog / G1 robots, Linux) |
| `weights` | `followbench2_weights.zip` | 3.7 GB | 4.0 GB | perception baselines (R18 / KPR ReID, trajectory predictors) |
| `oa_vat` | `followbench2_oa_vat.zip` | 0.6 GB | 0.7 GB | `PLANNER=oa_vat` (ORTrack + MobileCLIP) |
| `trackvla` | `followbench2_trackvla.zip` | 16.3 GB | 19.8 GB | `PLANNER=trackvla` (checkpoint + Qwen3-4B + SigLIP) |

`core` = `carla` + `weights`. `all` = all four.

```bash
bash tools/install/install_assets.sh --download --install-wheel core   # CARLA + perception weights
bash tools/install/install_assets.sh --download all                    # + OA-VAT + TrackVLA
```

What the installer does, per package:

1. checks free disk space;
2. downloads the zip into `downloads/` with `curl` (resumable: re-run the same
   command after an interruption);
3. checks the zip's size and SHA256 against
   [`tools/install/assets.tsv`](../tools/install/assets.tsv);
4. unpacks CARLA to `CARLA_multi_robots/` and the weights to `data/`;
5. for `weights` / `oa_vat`, downloads YOLO11 / YOLOe from their upstream
   publishers and checks their SHA256 (these are AGPL-3.0 and not redistributed
   in our zips);
6. runs `check_install.py` on the result.

Packages that are already installed are skipped. Options:

| Option / variable | Effect |
|---|---|
| `--download` | Download missing zips. Without it the zips must already be in `--zip-dir` |
| `--zip-dir DIR` | Read / download zips from `DIR` instead of `downloads/` |
| `--carla-dir DIR` | Install CARLA elsewhere; then `export CARLA_ROOT=DIR` |
| `--install-wheel` | `pip install` the CARLA client wheel of the build into the active env |
| `--delete-zips` | Remove each zip once it is verified and unpacked |
| `--force` | Re-extract packages that are already installed |
| `--check` | Only run the checker for the given packages |
| `HF_ENDPOINT` | Hugging Face endpoint, e.g. `https://hf-mirror.com` if `huggingface.co` is blocked |
| `FB_ASSETS_REVISION` | Dataset revision (branch, tag or commit) to download, default `main` |

To keep the weights on another disk, create the link first:
`ln -s /big/disk/followbench_data data`.

**Manual download.** You can also fetch the zips yourself, e.g.

```bash
pip install -U huggingface_hub
hf download hjyeee/Follow-Bench-2.0-packages --repo-type dataset --local-dir downloads
bash tools/install/install_assets.sh core      # verifies and unpacks what is in downloads/
```

Every launcher looks for CARLA in this order: `$CARLA_BIN`,
`$CARLA_ROOT/CarlaUE4.sh`, then `<repo>/CARLA_multi_robots/CarlaUE4.sh`. With the
default install location nothing needs to be exported.

## 3. Check the install

```bash
python tools/install/check_install.py core       # or: oa_vat, trackvla, all
python tools/install/check_install.py --deep all # SHA256 of every file (~40 GB read)
```

Every file the benchmark loads is listed with its size and SHA256 in
[`tools/install/files.tsv`](../tools/install/files.tsv): the files inside the
zips, the upstream YOLO weights and the gated DINOv3 checkpoints. By default
the checker compares the size of every file and the SHA256 of files under
128 MiB; `--deep` hashes all of them. It also checks the active Python env:
`followbench` / `baselines` are imported from this checkout, the CARLA client
version equals the build's `VERSION`, and torch sees a CUDA device.

The exit code is 0 only if everything in the requested scope passes. Each
failure is printed with the command that fixes it, for example:

```text
[dinov3_trackvla] 0/3 files ok  -> FAIL (required)
  MISSING .../data/trackvla/facebook/dinov3-vits16-pretrain-lvd1689m/model.safetensors
  fix: gated on Hugging Face; see docs/01_install.md, 'DINOv3 for TrackVLA'
```

Scopes:

| Scope | Files | Enough for |
|---|---|---|
| `core` | CARLA, `data/weights/`, YOLO11 | every planner except `trackvla` / `oa_vat` |
| `oa_vat` | `data/oa-vat/`, YOLOe, DINOv3 ViT-B/16 `.pth` | `PLANNER=oa_vat` |
| `trackvla` | `data/trackvla/`, DINOv3 ViT-S/16 HF snapshot | `PLANNER=trackvla` |
| `all` | all of the above | the full baseline table |

## 4. Start CARLA

```bash
bash tools/carla/run_carla.sh -RenderOffScreen -nosound -quality-level=Low
# multi-GPU / custom port example:
#   CUDA_VISIBLE_DEVICES=3 CARLA_multi_robots/CarlaUE4.sh \
#     -RenderOffScreen -nosound -quality-level=Low \
#     -graphicsadapter=<vulkan_index> -carla-rpc-port=2300
```

Batch and multi-GPU scripts start and stop CARLA themselves. The Vulkan GPU
index may differ from the `nvidia-smi` index; see `docs/07_troubleshooting.md`.

## 5. Weights layout

After step 2 the repository contains:

```text
data/weights/                  # followbench2_weights.zip
  yolo/                        # yolo11n.pt, yolo11s.pt  (fetched by the installer)
  reid_basic/                  # ckpt.t7            (R18 ReID)
  reid_kpr/                    # kpr_*.pth.tar      (KPR ReID) + LICENSE
  traj_predictor/              # sgan.pt, csgan.pt, csgan2.pt
data/oa-vat/                   # followbench2_oa_vat.zip
  ortrack/  mobileclip_blt.ts  LICENSE_MODELS
  yolo/yoloe-11l-seg.pt        # fetched by the installer
  dinov3/                      # you download the .pth (gated)
data/trackvla/                 # followbench2_trackvla.zip
  ckpts/<run>/model_epoch04_step024000.pt
  LLM_hf/qwen3-4b/  google/siglip-so400m-patch14-384/
  facebook/dinov3-vits16-pretrain-lvd1689m/   # you download the snapshot (gated)
```

TrackVLA's optional auto-caption model (`data/trackvla/LLM_hf/qwen3-vl-4b-instruct`,
Qwen3-VL-4B-Instruct from Hugging Face) is not included. Without it, TrackVLA
uses the plain instruction "Follow the person".

The KPR code finds its weights through a tracked symlink:

```text
followbench/perception/reid_kpr/deep_person_reid/pretrained_models
  → ../../../../data/weights/reid_kpr
```

### Weights you download yourself

These checkpoints are not in our zips. `check_install.py` lists the ones still
missing and checks the SHA256 of the ones you add.

**YOLO11 and YOLOe** (AGPL-3.0). Fetched and verified automatically by
`install_assets.sh` (from the `Ultralytics/YOLO11` and `jameslahm/yoloe`
Hugging Face repos, with the Ultralytics GitHub release as fallback). Manual
equivalent:

```bash
curl -L -o data/weights/yolo/yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
curl -L -o data/weights/yolo/yolo11s.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
curl -L -o data/oa-vat/yolo/yoloe-11l-seg.pt https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11l-seg.pt
```

License: <https://github.com/ultralytics/ultralytics/blob/main/LICENSE>.

<a id="dinov3-for-trackvla"></a>
**DINOv3 for TrackVLA** (only for `PLANNER=trackvla`). The Hugging Face repo is
gated. Accept the license at
<https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m>, then:

```bash
hf auth login
hf download facebook/dinov3-vits16-pretrain-lvd1689m \
  --local-dir data/trackvla/facebook/dinov3-vits16-pretrain-lvd1689m
```

<a id="dinov3-for-oa-vat"></a>
**DINOv3 for OA-VAT** (only for `PLANNER=oa_vat`). The code loads `.pth`
files, not the Hugging Face safetensors. Request access at
<https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>.
After the approval email, download with the link from that email and save it as:

| File under `data/oa-vat/dinov3/` | Role |
|---|---|
| `dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth` | default backbone (checked by `check_install.py oa_vat`) |
| `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` | only for `--oa-vat-dinov3 dinov3_vits16` |

The 8 hex digits in each name are the start of the file's SHA256.

### Licenses for weights that are in the zips

- **Qwen3-4B** — Apache-2.0. `data/trackvla/LLM_hf/qwen3-4b/LICENSE`.
- **SigLIP so400m-patch14-384** — Apache-2.0. The Hugging Face snapshot does
  not ship a LICENSE file, so the zip includes the Apache-2.0 text from
  <https://github.com/google-research/big_vision> at
  `data/trackvla/google/siglip-so400m-patch14-384/LICENSE`.
- **ORTrack** — MIT, Copyright (c) 2025 You Wu.
  `data/oa-vat/ortrack/LICENSE`.
  <https://github.com/wuyou3474/ORTrack>
- **KPR** — Hippocratic License 3.0 (HL3-LAW-MEDIA-MIL-SOC-SV).
  `data/weights/reid_kpr/LICENSE` (full text) and `NOTICE.md`.
  Redistribution is allowed and derivative works must keep this license.
  The ethical-use conditions restrict mass surveillance, military
  activities, providing goods or services to law enforcement, and
  broadcasting extreme violence, among other things. Read the license.
  <https://github.com/VlSomers/keypoint_promptable_reidentification>
- **MobileCLIP-B-LT** (`data/oa-vat/mobileclip_blt.ts`) — Apple Machine
  Learning Research Model. `LICENSE_MODELS` and `MOBILECLIP_NOTICE.txt`
  are in `data/oa-vat/`. Use is limited to non-commercial scientific
  research. Commercial use is outside that agreement.

  Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.

  <https://github.com/apple/ml-mobileclip>

## 6. Sanity check

```bash
python tools/install/check_install.py core
python tests/test_target_reid.py && python tests/test_perception_states.py   # offline, no CARLA
FB_DRY_RUN=1 bash scripts/run_episode.sh --scene corner                      # prints the episode command
```

Then follow the README "Quick start" to run an episode and a small batch.
