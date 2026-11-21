"""REPA-style patch alignment for Priming and RePrime.

The head projects intermediate patch tokens through a small MLP and maximizes
their cosine similarity with a frozen encoder's target tokens. It is used in
two ways:

* Pretraining (RePrime): ``target='self'`` and the caller passes the clean image
  ``x_0`` for each corrupted ``x_s``, so the target is instance-paired.
  This is the placement that the paper recommends.
* Post-training (placement study): the target can be a random same-class image,
  a pooled class set, the image selected by the OT plan, the generated sample
  itself, or the EMA teacher's sample.

The projection head is training-only and is discarded at inference. The
alignment term can be windowed with ``start_step`` / ``end_step`` /
``ramp_steps``; the OT objective itself is unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OTGuidedAlignment(nn.Module):
    """REPA-style patch alignment with paired, pooled, self, EMA, or OT-plan targets."""

    def __init__(self, *, hidden_size: int, block: int = 8, weight: float = 1.0,
                 start_step: int = 250, ramp_steps: int = 100, end_step: int = 2500,
                 pairing: str = 'argmax', target: str = 'plan_paired',
                 loss_form: str = 'maxcos', tokenwise_interp: bool = False,
                 feature_extractor: str = 'dinov2', ot_encoder=None,
                 ot_family: str = 'convnext',
                 ot_token_key: str = 'convenxt_stage_2',
                 model_name: str = 'vit_large_patch14_dinov2.lvd142m',
                 img_size: int = 256, device='cuda'):
        super().__init__()
        if pairing not in {'argmax', 'sample'}:
            raise ValueError(f'unknown pairing {pairing!r}')
        if target not in {'plan_paired', 'random_paired', 'pooled', 'self', 'ema'}:
            raise ValueError(f'unknown target mode {target!r}')
        if feature_extractor not in {'dinov2', 'ot_stack'}:
            raise ValueError(f'unknown feature_extractor {feature_extractor!r}')
        if loss_form not in {'maxcos', 'tokenwise'}:
            raise ValueError(f'unknown loss_form {loss_form!r}')
        if feature_extractor == 'ot_stack':
            if ot_encoder is None:
                raise ValueError('ot_stack targets require the frozen OT encoder')
            if ot_family not in {'convnext', 'mae'}:
                raise ValueError(f'unknown ot_family {ot_family!r}')
        self.target = target
        self.loss_form = loss_form
        self.tokenwise_interp = bool(tokenwise_interp)
        self.pool_size = 16
        self.feature_extractor = feature_extractor
        self.ot_family = ot_family
        self.ot_token_key = ot_token_key
        self.ot_encoder = ot_encoder if feature_extractor == 'ot_stack' else None
        self.img_size = int(img_size)
        self.device = torch.device(device)
        self.block = int(block)
        self.weight = float(weight)
        self.start_step = int(start_step)
        self.ramp_steps = max(0, int(ramp_steps))
        self.end_step = int(end_step)
        self.pairing = pairing
        self.img_size = int(img_size)
        # DINOv2 patch 14 requires an input size divisible by 14: resize to the
        # nearest multiple (256 -> 252, 18x18 = 324 tokens).
        self.patch_size = 14
        self.target_size = self.patch_size * max(1, self.img_size // self.patch_size)
        self.device = torch.device(device)

        if self.feature_extractor == 'dinov2':
            import timm
            from timm.data import resolve_data_config
            # Deterministic init so a resumed run rebuilds the same head weights.
            state = torch.random.get_rng_state()
            try:
                torch.manual_seed(20261020)
                encoder = timm.create_model(model_name, pretrained=True, num_classes=0,
                                            img_size=self.img_size, dynamic_img_size=True)
            finally:
                torch.random.set_rng_state(state)
            encoder = encoder.to(self.device).eval().requires_grad_(False)
            data_config = resolve_data_config(encoder.pretrained_cfg, model=encoder)
            self.register_buffer(
                'mean', torch.tensor(data_config['mean'], dtype=torch.float32,
                                     device=self.device).view(1, 3, 1, 1),
                persistent=False)
            self.register_buffer(
                'std', torch.tensor(data_config['std'], dtype=torch.float32,
                                    device=self.device).view(1, 3, 1, 1),
                persistent=False)
            self.encoder = encoder
            target_dim = int(encoder.num_features)
        else:
            # Reuse the frozen OT encoders: probe the token dimension once.
            self.encoder = None
            self.register_buffer('mean', torch.zeros(1, 3, 1, 1), persistent=False)
            self.register_buffer('std', torch.ones(1, 3, 1, 1), persistent=False)
            with torch.no_grad():
                probe = self.encode_targets(
                    torch.zeros(2, 3, self.img_size, self.img_size, device=self.device))
            target_dim = int(probe.shape[-1])
        # Seeded head init (deterministic across restarts for resume safety).
        state = torch.random.get_rng_state()
        try:
            torch.manual_seed(20261020)
            head = nn.Sequential(
                nn.Linear(int(hidden_size), target_dim),
                nn.GELU(),
                nn.Linear(target_dim, target_dim),
            )
        finally:
            torch.random.set_rng_state(state)
        self.proj = head.to(self.device)
        # Safety net: move every parameter/buffer (encoder weights, proj, norm
        # statistics) onto the target device exactly once.
        self.to(self.device)
        self._hidden = None
        self._handle = None

    # ---------------------------------------------------------------- hooks
    def attach(self, model):
        """Hook the requested backbone block of ``model.net`` (Priming wrapper)."""
        blocks = model.net.blocks
        if not 0 <= self.block < len(blocks):
            raise ValueError(f'block {self.block} outside [0,{len(blocks)})')
        self._handle = blocks[self.block].register_forward_hook(self._capture)
        return self

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _capture(self, module, inputs, output):
        self._hidden = output

    def pop_hidden(self):
        hidden, self._hidden = self._hidden, None
        return hidden

    # -------------------------------------------------------------- targets
    @torch.no_grad()
    def encode_targets(self, images):
        if self.feature_extractor == 'ot_stack':
            # Mirror models/encoders/full/encoders.py family preprocessing and
            # return spatial tokens of the requested calibrated layer.
            x = images.float()
            if self.ot_family == 'convnext':
                mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
                std = x.new_tensor([.229, .224, .225])[None, :, None, None]
                pixels = ((x - mean) / std).permute(0, 2, 3, 1).contiguous()
                with torch.autocast(x.device.type, enabled=False):
                    rows = self.ot_encoder.encoders.convnext.get_activations(pixels)
            else:
                pixels = (x * 2 - 1).permute(0, 2, 3, 1).contiguous()
                mixed = bool(self.ot_encoder.encoders.mixed_precision)
                with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=mixed):
                    rows = self.ot_encoder.encoders.mae.get_activations(pixels)
            features = rows[self.ot_token_key]
            if features.ndim != 3:
                raise RuntimeError('unexpected OT-encoder token rank')
            return features
        x = images.float()
        if x.shape[-1] != self.target_size or x.shape[-2] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                              mode='bicubic', align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        features = self.encoder.forward_features(x)
        if features.ndim != 3:
            raise RuntimeError('unexpected target-encoder output rank')
        tokens = (self.target_size // self.patch_size) ** 2
        if features.shape[1] != tokens:
            features = features[:, -tokens:]
        return features

    # ----------------------------------------------------------------- loss
    def forward(self, hidden, target_images, plan=None, plan_weights=None, *,
                prefix_tokens, step):
        if step < self.start_step:
            return None
        if self.end_step and step >= self.end_step:
            return None
        lam = self.weight
        if self.ramp_steps > 0:
            lam *= min(1.0, (step - self.start_step + 1) / self.ramp_steps)
        if lam <= 0:
            return None
        patches = hidden[:, int(prefix_tokens):]
        projected = F.normalize(self.proj(patches.float()), dim=-1)    # [B, L, D]
        extra = {}
        classes = int(target_images.shape[0])
        references = int(target_images.shape[1])
        generated = patches.shape[0] // classes
        if generated * classes != patches.shape[0]:
            raise RuntimeError(
                f'hidden batch {patches.shape[0]} is not classes*generated')
        flat = target_images.reshape(-1, *target_images.shape[-3:])
        if self.target == 'random_paired':
            # V1: faithful REPA in stage 2 -- pair each generated sample with a
            # uniformly random real image of the same class (no OT guidance).
            index = torch.randint(0, references, (classes, generated),
                                  device=flat.device)
            offsets = torch.arange(classes, device=flat.device).repeat_interleave(
                generated) * references
            global_index = index.reshape(-1) + offsets
            targets = self.encode_targets(flat[global_index])           # [B, P, D]
            extra = dict(paired_columns=index.detach(),
                         paired_rows=global_index.detach())
        elif self.target == 'pooled':
            # V2: no per-sample pairing -- every generated patch matches the
            # best token over a subset of the class's real-image pool.
            features = self.encode_targets(flat)                        # [C*R, P, D]
            features = features.reshape(classes, references,
                                        features.shape[1], features.shape[2])
            k = min(self.pool_size, references)
            pick = torch.randperm(references, device=features.device)[:k]
            pool = F.normalize(
                features[:, pick].reshape(classes, k * features.shape[2],
                                          features.shape[3]).float(), dim=-1)
            projected = projected.reshape(classes, generated, -1,
                                          projected.shape[-1])          # [C, G, L, D]
            cosine_sum, token_count = 0.0, 0
            chunk = max(1, min(16, generated))
            for start in range(0, generated, chunk):
                part = projected[:, start:start + chunk]                # [C, g, L, D]
                sim = torch.einsum('cgld,csd->cgls', part, pool)        # [C,g,L,S]
                cosine_sum = cosine_sum + sim.max(-1).values.sum()
                token_count += classes * part.shape[1] * part.shape[2]
            mean_best = cosine_sum / max(1, token_count)
            loss = 1.0 - mean_best
            return dict(loss=loss, weight=lam, cosine=mean_best.detach(),
                        pool_images=torch.as_tensor(k, device=flat.device))
        elif self.target == 'plan_paired':
            # V3: pair every generated sample with a real image through the OT plan.
            if plan is None or plan_weights is None:
                raise RuntimeError('plan_paired target requires the OT plan')
            weights = plan_weights.to(plan.device).view(-1, 1, 1, 1)
            pooled = (weights * plan.float()).sum(0)                   # [C, G, R]
            conditional = pooled / pooled.sum(-1, keepdim=True).clamp_min(1e-12)
            classes, generated, references = conditional.shape
            expected = classes * generated
            if patches.shape[0] != expected:
                raise RuntimeError(
                    f'captured hidden batch {patches.shape[0]} != classes*generated {expected}')
            if self.pairing == 'argmax':
                index = conditional.argmax(-1)                         # [C, G] in [0,R)
            else:
                index = torch.multinomial(conditional.reshape(-1, references), 1
                                         ).reshape(classes, generated)
            offsets = torch.arange(classes, device=index.device).repeat_interleave(
                generated) * references
            global_index = index.reshape(-1) + offsets
            targets = self.encode_targets(flat[global_index])          # [B, P, D]
            extra = dict(paired_columns=index.detach(), paired_rows=global_index.detach())
        else:
            # V4a ('self'): targets are the generated images themselves.
            # V4b ('ema'): targets are the EMA teacher's generations for the same noise.
            if patches.shape[0] != flat.shape[0]:
                raise RuntimeError(
                    f'captured hidden batch {patches.shape[0]} != target images {flat.shape[0]}')
            targets = self.encode_targets(flat)                        # [B, P, D]
        target = F.normalize(targets.float(), dim=-1)
        if self.loss_form == 'maxcos':
            similarity = torch.einsum('bld,bpd->blp', projected, target)  # [B, L, P]
            best = similarity.max(-1).values
        else:
            # Token-wise (REPA) alignment: h[b, t] <-> f[b, t] at the same grid
            # cell. Target token grids are resampled to the source grid when
            # requested (e.g. DINOv2 18x18 -> 16x16, ConvNeXt 14x14 -> 16x16).
            if target.shape[1] != projected.shape[1]:
                if not self.tokenwise_interp:
                    raise RuntimeError(
                        'tokenwise loss needs matching token grids; got '
                        f'{projected.shape[1]} vs {target.shape[1]}')
                source_tokens = int(projected.shape[1])
                target_tokens = int(target.shape[1])
                grid = int(round(source_tokens ** 0.5))
                side = int(round(target_tokens ** 0.5))
                if grid * grid != source_tokens or side * side != target_tokens:
                    raise RuntimeError('tokenwise_interp needs square token grids')
                b, _, d = target.shape
                resampled = target.reshape(b, side, side, d).permute(0, 3, 1, 2)
                resampled = F.interpolate(resampled, size=(grid, grid),
                                          mode='bilinear', align_corners=False)
                target = F.normalize(
                    resampled.permute(0, 2, 3, 1).reshape(b, grid * grid, d), dim=-1)
            best = (projected * target).sum(-1)                        # [B, L]
        loss = 1.0 - best.mean()
        return dict(loss=loss, weight=lam, cosine=best.mean().detach(), **extra)


def make_alignment(config, model, device='cuda', ot_encoder=None):
    """Build and attach the alignment module, or return None when disabled."""
    section = config.get('alignment') or {}
    if not section.get('enabled', False):
        return None
    if section.get('method', 'ot_guided') != 'ot_guided':
        raise ValueError(f"unsupported alignment method {section.get('method')!r}")
    feature_extractor = str(section.get('feature_extractor', 'dinov2'))
    alignment = OTGuidedAlignment(
        hidden_size=int(model.net.hidden_size),
        block=int(section.get('block', 8)),
        weight=float(section.get('weight', 1.0)),
        start_step=int(section.get('start_step', 250)),
        ramp_steps=int(section.get('ramp_steps', 100)),
        end_step=int(section.get('end_step', 2500)),
        pairing=str(section.get('pairing', 'argmax')),
        target=str(section.get('target', 'plan_paired')),
        loss_form=str(section.get('loss_form', 'maxcos')),
        tokenwise_interp=bool(section.get('tokenwise_interp', False)),
        img_size=int(section.get('img_size', 256)),
        feature_extractor=feature_extractor,
        ot_encoder=ot_encoder,
        ot_family=str(section.get('ot_family', 'convnext')),
        ot_token_key=str(section.get('ot_token_key', 'convenxt_stage_2')),
        model_name=str(section.get('target_encoder', 'vit_large_patch14_dinov2.lvd142m')),
        device=device,
    ).attach(model)
    return alignment
