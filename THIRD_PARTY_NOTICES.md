# Third-Party Notices

FollowBench 2.0 includes source code, model weights, and binary assets from
third-party projects. The project itself is released under the MIT License
(see [`LICENSE`](LICENSE)); that license covers only the FollowBench 2.0
authors' own contributions and does **not** override the licenses of the
third-party components listed below.

For every component we state:

- where it lives in this repository,
- the upstream project it comes from,
- the applicable license, and
- where the full license text is kept (under `licenses/`, or inside the
  bundled directory itself).

Full license texts for the non-standard licenses are stored in `licenses/`.
The standard permissive licenses (MIT, BSD-3-Clause, Apache-2.0) are referenced
by their SPDX identifier and canonical text.

---

## Bundled code

### OA-VAT (`baselines/third_party/oa-vat/`)

- Upstream: <https://github.com/SHWplus/OA-VAT>
- The upstream repository does not publish a license file. The code is
  reproduced here as part of the OA-VAT baseline adapter; copyright remains
  with the OA-VAT authors.
- Sub-components inside this directory have their own licenses:
  - `ORTrack/` — <https://github.com/wuyou3474/ORTrack>, MIT License,
    Copyright (c) 2025 You Wu. Full text: `licenses/LICENSE_ORTrack_MIT.txt`.
    It embeds Vision-Transformer / DeiT / EVA backbone code derived from
    [timm](https://github.com/huggingface/pytorch-image-models) (Apache-2.0),
    [facebookresearch/deit](https://github.com/facebookresearch/deit)
    (Apache-2.0), and [baaivision/EVA](https://github.com/baaivision/EVA)
    (MIT). `timm` is also a runtime dependency.
  - `kalman_filter.py` — 8-state image-space Kalman filter as published by
    [ByteTrack](https://github.com/ifzhang/ByteTrack) (MIT License).
  - `dinov3-main/` — <https://github.com/facebookresearch/dinov3>, DINOv3
    License. Full text: `licenses/LICENSE_DINOv3.txt`.
  - The OpenAI CLIP tokenizer used by the YOLOe text encoder is MIT
    (Copyright (c) 2021 OpenAI); full text:
    `licenses/LICENSE_OpenAI_CLIP_MIT.txt`.

### TrackVLA (`baselines/third_party/trackvla/`)

- `socialRPF/model/cache_gridpool.py` — from
  <https://github.com/om-ai-lab/OmTrackVLA> (formerly OpenTrackVLA). The
  upstream repository does not publish a license file; copyright remains with
  the OmTrackVLA authors.

### KPR ReID (`followbench/perception/reid_kpr/deep_person_reid/`)

- Upstream: <https://github.com/VlSomers/keypoint_promptable_reidentification>
  (KPR), a fork of
  [deep-person-reid / torchreid](https://github.com/KaiyangZhou/deep-person-reid).
- KPR is licensed under the **Hippocratic License 3.0** (HL3-LAW-MEDIA-MIL-SOC-SV).
  Full text: `licenses/LICENSE_KPR_Hippocratic-3.0.txt`. Please read the ethical-use
  conditions before use.
- The bundled `torchreid/` library is MIT, Copyright (c) 2018 Kaiyang Zhou.
  Full text: `licenses/LICENSE_torchreid_MIT.txt`.
- Backbone implementations bundled under `torchreid/models/`:
  - `transreid/` — <https://github.com/damo-cv/TransReID>, MIT License.
  - `solider/` — <https://github.com/tinyvision/SOLIDER>, Apache-2.0.
    Its `swin_transformer.py` derives from
    [microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer)
    (MIT; full text `licenses/LICENSE_Swin-Transformer_MIT.txt`).
  - `pvpm.py` — <https://github.com/hh23333/PVPM>. The upstream repository does
    not publish a license file; copyright remains with the PVPM authors.
  - `hrnet.py` — based on
    [HRNet/HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification)
    (MIT) and [CASIA-IVA-Lab/ISP-reID](https://github.com/CASIA-IVA-Lab/ISP-reID)
    (Apache-2.0).

### RDA planner (`baselines/planning/rda/`)

- `core/rda_solver.py` and related MPC code — Author: Han Ruihua,
  <https://github.com/hanruihua/RDA-planner>, MIT License,
  Copyright (c) 2023 Ruihua Han. Full text:
  `licenses/LICENSE_RDA-planner_MIT.txt`.

### Trajectory prediction (`baselines/prediction/`)

- `utils/sgan_models.py`, `utils/sgan_utils.py` — based on
  [Social GAN](https://github.com/agrimgupta92/sgan), MIT License,
  Copyright (c) 2018 Agrim Gupta, Justin Johnson. Full text:
  `licenses/LICENSE_SocialGAN_MIT.txt`.
- `backends/sgan.py`, `backends/cvkf.py`, `backends/cv.py`,
  `utils/optimized_sgan.py` — adapted from
  [ComplexityNav](https://github.com/fluentrobotics/ComplexityNav); the
  predictor implementations themselves originate from
  [Pred2Nav](https://github.com/sriyash421/Pred2Nav), BSD 3-Clause License,
  Copyright (c) 2026 Sriyash Poddar. Full text:
  `licenses/LICENSE_Pred2Nav_BSD-3-Clause.txt`.

### Deep SORT (ReID basic checkpoint lineage)

- `data/weights/reid_basic/ckpt.t7` — ReID checkpoint; the network
  architecture in `followbench/perception/reid_model/net.py` is a compact
  ResNet-style backbone trained on Market-1501. See
  [ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch)
  (MIT License) for the original model/checkpoint lineage. Full text:
  `licenses/LICENSE_deep_sort_pytorch_MIT.txt`.

---

## Distributed model weights

These weights are not stored in this git repository; they are fetched by
`tools/install/install_assets.sh` (or downloaded manually). Their license files
ship inside the release zips.

| Component | Location (after install) | License |
|---|---|---|
| KPR ReID checkpoints | `data/weights/reid_kpr/` | Hippocratic License 3.0 (`LICENSE`, `NOTICE.md` in that dir) |
| ORTrack checkpoint | `data/oa-vat/ortrack/` | MIT (`data/oa-vat/ortrack/LICENSE`) |
| MobileCLIP-B-LT | `data/oa-vat/mobileclip_blt.ts` | Apple Machine Learning Research Model License (`data/oa-vat/LICENSE_MODELS`) — non-commercial research only |
| Qwen3-4B | `data/trackvla/LLM_hf/qwen3-4b/` | Apache-2.0 (`LICENSE` in that dir) |
| SigLIP so400m-patch14-384 | `data/trackvla/google/siglip-so400m-patch14-384/` | Apache-2.0 (`LICENSE`, `NOTICE` in that dir) |
| DINOv3 (TrackVLA / OA-VAT) | `data/trackvla/…`, `data/oa-vat/dinov3/…` | DINOv3 License (gated download) |
| CARLA 0.9.16 build | `CARLA_multi_robots/` | code MIT; assets CC-BY; binary contains Unreal Engine components under the UE EULA |
| YOLO11 / YOLOe | `data/weights/yolo/…`, `data/oa-vat/yolo/…` | AGPL-3.0 — downloaded from upstream at install time, not redistributed |

See `docs/01_install.md` for the complete per-file list, SHA256 values, and
download instructions.
