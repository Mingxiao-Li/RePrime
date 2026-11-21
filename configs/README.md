# Configuration map

Every recipe inherits its shared settings from `_base.yaml` in the same
directory. Paths are relative to the repository root; `data/`, `assets/`, and
`checkpoints/` are intentionally not committed. Override them with
`--data-path`, `--encoder-cache`, `--hf-home`, `--torch-home`, and
`--init-checkpoint` (see the root README).

## Main paper experiments

| Paper result | Pretraining config | Post-training config | Notes |
|---|---|---|---|
| Priming-B / 160 | `pretrain/priming_b_160.yaml` | `posttrain/priming_b_160.yaml` | baseline |
| Priming-L / 160 | `pretrain/priming_l_160.yaml` | `posttrain/priming_l_160.yaml` | baseline |
| **RePrime-B / 160** | `pretrain/reprime_b_160.yaml` | `posttrain/reprime_b_160.yaml` | block 5, lambda 0.5, DINOv2-L/14, full window |
| **RePrime-B / 160, block 8** | `pretrain/reprime_b_160_blk8.yaml` | `posttrain/reprime_b_160.yaml` | best B result |
| **RePrime-L / 160** | `pretrain/reprime_l_160.yaml` | `posttrain/reprime_l_160.yaml` | block 10 of 32 |
| **RePrime-J / 160** | `pretrain/reprime_joint_b_160.yaml` | optional `posttrain/reprime_b_160.yaml` | native model; refine with 25k OT updates |
| RePrime-B / 200, block 8 | `pretrain/reprime_b_200_blk8.yaml` | `posttrain/reprime_b_160.yaml` | budget-curve arm |

Post-training recipes are the official 25,000-update OT recipe. The only
difference between `priming_*` and `reprime_*` is the initialization checkpoint,
passed with `--init-checkpoint`; the recipe itself is unchanged.

## Ablations (40-epoch protocol unless noted)

| Paper question | Config |
|---|---|
| Teacher: DINOv2 vs ResNet-MAE | `pretrain/abl_teacher_mae.yaml` (160 epochs; compare with `reprime_b_160.yaml`) |
| Teacher: DINOv2 vs ConvNeXt | `pretrain/abl_teacher_convnext.yaml` (160 epochs) |
| Depth: block 2 / 8 / 11 | `pretrain/abl_depth_blk2.yaml`, `abl_depth_blk8.yaml`, `abl_depth_blk11.yaml` |
| Window: first 25% / 50% / 100% | `pretrain/abl_window_25.yaml`, `abl_window_50.yaml`, `abl_window_100.yaml` |
| Correspondence control | `pretrain/abl_corr_self.yaml`, `abl_corr_random.yaml`, `abl_corr_permuted.yaml` |
| Placement: generative path | `posttrain/placement/*.yaml` |
| Placement: paired path | `posttrain/placement/b160_repapair.yaml` |

The depth/window recipes use the 40-epoch protocol and `train.allow_short: true`; this
is a sanctioned shortened protocol for ablations, not a change to the
160-epoch paper recipe.

## Placement / generative-path configs

`posttrain/placement/` contains the stage-2 alignment configurations used for
the placement study (random same-class, OT-matched, pooled, self, EMA teacher,
feature-space variants, and the paired-path `b160_repapair.yaml`). They inherit
`../_base.yaml` and are intended for the 2.5k-update protocol described in the
paper, not for the 25k main runs.

## Hardware

* B pretraining and B post-training: 8x H100 80GB or H200.
* L post-training: 8x H200 (the original runs exceed 85 GiB allocated per GPU).
* Paper FID/FDr5 evaluation: 4 ranks, batch 32, 50,000 images, 3 seeds.
