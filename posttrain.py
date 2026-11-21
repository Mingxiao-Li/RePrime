"""Priming post-training: host-owned sampling, AdamW/EMA, and scheduled evaluation."""
import argparse
from functools import partial
import json
import logging
import os
from pathlib import Path
import time

from utils.paths import ROOT, configure_environment, resolve as resolve_path

resolve = partial(resolve_path, canonical=True)
LOG = logging.getLogger(__name__)


def make_matcher(cache, *, stack='convnext_full_mae', device='cuda', microbatch=8,
                 hf_home=None, torch_home=None, **objective):
    from losses.matching import Matching
    from models.encoders.load import FrozenEncoders
    if stack=='sim' and objective.get('kind','featurewise_ot')!='featurewise_ot':
        raise ValueError('The SIM recipe uses featurewise OT')
    encoder = FrozenEncoders(cache, stack=stack, device=device, microbatch=microbatch,
                             hf_home=hf_home, torch_home=torch_home)
    return Matching(encoder, bandwidths=encoder.bandwidths, weights=encoder.weights,
                    checkpoint_metadata=encoder.checkpoint_metadata(), **objective)


def backward_rollout(model, matcher, reference, stream, batch, noise, sampling, *,
                     noise_scale=1., amp=True, step=None, alignment=None, ema=None):
    import torch
    from utils.collectives import sync_gradients

    labels = batch.generated_labels
    if len(noise)!=len(labels) or noise.device!=labels.device:
        raise ValueError('Noise and selected class labels must share batch size and device')
    classes, samples = matcher.cross.local_classes_per_step, matcher.cross.samples_per_class
    real = reference.fetch(batch.target_indices, batch.target_labels).to(labels.device, non_blocking=True)

    teacher_images = None
    if alignment is not None and alignment.target == 'ema':
        if ema is None:
            raise RuntimeError('ema target requires the EMA shadow')
        # Same noise and labels, EMA weights swapped in temporarily (no grad).
        with torch.no_grad(), torch.autocast(labels.device.type, dtype=torch.bfloat16, enabled=amp):
            with ema.swap(model):
                teacher = model.sample_images_with_grad(noise, labels,
                                                        sampling_args=dict(sampling))
            teacher = teacher*.5+.5
        teacher_images = teacher.reshape(classes, samples, *teacher.shape[1:])

    def generate(value):
        # 2-GPU adaptation (env-gated): with <=2 ranks each rank generates 512
        # images and the autograd graph exceeds 64GB-class GPUs. PRIMING_CKPT_GEN=1
        # enables activation checkpointing (math unchanged; recomputes activations
        # in backward). The official 8-rank recipe leaves it off.
        with torch.autocast(value.device.type, dtype=torch.bfloat16, enabled=amp):
            if os.environ.get('PRIMING_CKPT_GEN', '0') == '1':
                from torch.utils.checkpoint import checkpoint
                images = checkpoint(model.sample_images_with_grad, value, labels,
                                    sampling_args=dict(sampling), use_reentrant=False)
            else:
                images = model.sample_images_with_grad(value, labels,
                                                       sampling_args=dict(sampling))
            images = images*.5+.5
        return images.reshape(classes, samples, *images.shape[1:])

    with torch.no_grad():
        negative_noise = torch.randn(len(labels), model.in_channels, model.input_size, model.input_size,
            device=labels.device, generator=stream.negative_noise_generator)*noise_scale
        x_prime = generate(negative_noise)
    x = generate(noise)
    y = real.reshape(classes, matcher.cross.real_references_per_class, *real.shape[1:])
    loss, metrics = matcher(x=x, x_prime=x_prime, y=y)
    if alignment is not None and step is not None:
        hidden = alignment.pop_hidden()
        plan = getattr(matcher, 'last_alignment_plan', None)
        plan_weights = getattr(matcher, 'last_alignment_weights', None)
        if alignment.target in ('plan_paired', 'random_paired', 'pooled'):
            target_images = y
        elif alignment.target == 'ema':
            target_images, plan, plan_weights = teacher_images, None, None
        else:  # self
            target_images, plan, plan_weights = x.detach(), None, None
        needs_plan = alignment.target == 'plan_paired'
        if hidden is not None and (not needs_plan or plan is not None):
            result = alignment(hidden, target_images, plan, plan_weights,
                               prefix_tokens=model.net.prefix_tokens, step=step)
            if result is not None:
                loss = loss + result['weight'] * result['loss']
                metrics = dict(metrics)
                metrics['align_loss'] = result['loss'].detach()
                metrics['align_weight'] = torch.as_tensor(
                    result['weight'], dtype=loss.dtype, device=loss.device)
                metrics['align_cosine'] = result['cosine']
    loss.backward()
    sync_gradients(model.parameters(), reduction='sum')
    return loss.detach(), metrics


def train_step(model, matcher, reference, stream, sampling, *, noise_scale=1., amp=True,
               step=None, alignment=None, ema=None):
    import torch
    batch = stream.next_batch()
    noise = torch.randn(len(batch.generated_labels), model.in_channels, model.input_size, model.input_size,
                        device=batch.generated_labels.device)*noise_scale
    return backward_rollout(model, matcher, reference, stream, batch, noise, sampling,
                            noise_scale=noise_scale, amp=amp, step=step,
                            alignment=alignment, ema=ema)


def optimizer_step(model, optimizer, ema, step, *, learning_rate=1e-5, warmup_steps=125, grad_clip=1.):
    """The historical zero-based schedule starts with LR zero at update zero."""
    import torch
    lr = learning_rate*step/warmup_steps if warmup_steps>0 and step<=warmup_steps else learning_rate
    for group in optimizer.param_groups:
        group['lr'] = lr
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite post-training gradient norm')
    optimizer.step()
    ema.step(model)
    optimizer.zero_grad(set_to_none=True)
    return norm, lr


def load_config(path):
    from utils.config import load_yaml
    config = load_yaml(resolve(path))
    model, train, matching = config['model'], config['train'], config['matching']
    if model['model'] not in {'OST_B', 'OST_L'}:
        raise ValueError('Post-training supports OST_B and OST_L')
    fixed_model = dict(img_size=256, patch_size=16, in_channels=3, label_drop_prob=0.,
        class_tokens=8, guidance_tokens=0, interval_tokens=0, norm_eps=.01, norm_p=1.,
        rope_2d=True, learned_pe=True, P_mean=.8, P_std=.8, perceptual_threshold=.8)
    if any(model.get(key) != value for key, value in fixed_model.items()):
        raise ValueError('Post-training model differs from the B/L production architecture')
    forensic = os.environ.get('PRIMING_FG_EVAL') == '1'
    cadences = (1250, 2500, 250) if forensic else (1250, 2500)
    horizons = ((25000, 1024, 'bf16', 1), (50000, 1024, 'bf16', 1))
    if (train['steps'], train['global_batch_size'], train['dtype'], train['seed']) not in horizons:
        raise ValueError('Production post-training uses 25,000 (or sanctioned 50,000 extension) updates')
    if train['save_every'] not in cadences:
        raise ValueError('Production post-training uses 25,000 updates, batch 1,024, BF16, seed 1, and saves every 1,250 updates')
    fixed_optimizer = dict(optimizer='adamw', lr=1e-5, beta1=.9, beta2=.95,
                           weight_decay=0., grad_clip=1., warmup_steps=125)
    if config['optimizer'] != fixed_optimizer:
        raise ValueError('Post-training requires the production fresh AdamW and zero-based 125-update warmup')
    fixed_matching = dict(coefficient=5., global_classes_per_step=16, samples_per_class=64,
        real_references_per_class=128, epsilon=.001, epsilon_mixture={'0.001':.75, '0.01':.25},
        num_iterations=1000, tolerance=1e-6, microbatch=8, seed=7919)
    if any(matching.get(key) != value for key, value in fixed_matching.items()):
        raise ValueError('Post-training class populations, calibration, or Sinkhorn settings differ')
    if matching['kind'] not in {'featurewise_ot', 'featurewise_energy', 'featurewise_laplacian_mmd'}:
        raise ValueError('Unsupported feature discrepancy')
    if matching['stack'] not in {'convnext_full_mae', 'sim'} or (matching['stack'] == 'sim' and matching['kind'] != 'featurewise_ot'):
        raise ValueError('SIM uses featurewise OT; energy/Laplacian use the full feature stack')
    if model['model'] == 'OST_L' and (matching['stack'], matching['kind']) != ('convnext_full_mae', 'featurewise_ot'):
        raise ValueError('The packaged L recipe uses full-feature OT')
    delta = .04549544751644135 if matching['kind'] == 'featurewise_ot' else 0.
    if matching['energy_delta'] != delta:
        raise ValueError('Historical energy calibration differs')
    if config['sampling'] != dict(cfg=1., t_min=.1, t_max=1., num_steps=1):
        raise ValueError('Production post-training uses one-step generation with CFG 1')
    evaluation = config['evaluation']
    if tuple(evaluation[key] for key in ('num_images', 'batch_size', 'prc_k', 'prc_batch_size')) != (50000, 64, 3, 10000) or evaluation['every'] not in cadences:
        raise ValueError('Evaluation requires 50,000 images per online/EMA draw every 1,250 updates')
    if config['data']['num_classes'] != 1000 or config['data']['reference_workers'] < 1 or train['print_freq'] < 1:
        raise ValueError('Expected 1,000 classes and positive worker/print counts')
    if not isinstance(train['compile'], bool):
        raise ValueError('train.compile must be true or false')
    for section, keys in (('data', ('data_path', 'encoder_cache')), ('evaluation', ('stats_dir', 'prc_reference')),
                          ('initialization', ('checkpoint',))):
        for key in keys:
            config[section][key] = str(resolve(config[section][key]))
    return config


def make_model(config, device='cuda'):
    from models import Priming
    kwargs = dict(config['model'])
    kwargs['backbone'] = kwargs.pop('model')
    return Priming(**kwargs, num_classes=config['data']['num_classes']).to(device)


def run_training(config, model, optimizer, ema, matcher, reference, stream, output, *,
                 completed=0, checkpoint=None, evaluate, stop=lambda: False, smoke_steps=None,
                 amp=True, alignment=None):
    """Save before evaluation; resume pending milestones before the next update."""
    import torch
    from utils.evaluation import scheduled_evaluation
    from utils.files import write_json
    from utils.posttrain_checkpoint import save_posttraining
    output = Path(output)
    horizon = config['train']['steps']
    if not 0 <= completed <= horizon:
        raise ValueError('Checkpoint step exceeds the training horizon')
    if smoke_steps is not None and smoke_steps < 1:
        raise ValueError('Smoke steps must be positive')
    limit = horizon if smoke_steps is None else min(horizon, completed+smoke_steps)
    last_saved = completed if checkpoint else None

    def save():
        path = output/'checkpoints'/f'step_{completed:06d}.pth'
        save_posttraining(path, model, optimizer, ema, matcher, stream, completed, config=config)
        if stream.rank == 0:
            write_json(output/'checkpoints/latest.json', dict(checkpoint=path.name, completed=completed))
        return path

    def draw():
        LOG.info('Evaluating online and EMA weights at update %s', completed)
        return scheduled_evaluation(output, checkpoint, completed, config, evaluate,
                                    rank=stream.rank, world=stream.world_size)

    if completed and completed % config['evaluation']['every'] == 0:
        if checkpoint is None:
            raise ValueError('A resumed evaluation milestone needs its saved checkpoint')
        draw()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for step in range(completed, limit):
        model.train()
        loss, metrics = train_step(model, matcher, reference, stream, config['sampling'],
                                   amp=amp, step=step, alignment=alignment, ema=ema)
        norm, lr = optimizer_step(model, optimizer, ema, step,
            learning_rate=config['optimizer']['lr'], warmup_steps=config['optimizer']['warmup_steps'],
            grad_clip=config['optimizer']['grad_clip'])
        completed = step+1
        if next(model.parameters()).device.type == 'cuda':
            torch.cuda.synchronize()
        if stream.rank == 0 and (completed == 1 or completed % config['train']['print_freq'] == 0):
            row = dict(step=completed, samples_seen=completed*stream.global_generated_count,
                       loss=float(loss), grad_norm=float(norm), lr=lr,
                       seconds=time.perf_counter()-started,
                       metrics={name: float(value) for name, value in metrics.items()})
            with (output/'training.jsonl').open('a') as handle:
                handle.write(json.dumps(row, allow_nan=False)+'\n')
            LOG.info('Update %s/%s loss=%.6g grad_norm=%.6g lr=%.6g', completed, horizon, loss, norm, lr)
        if completed % config['train']['save_every'] == 0 or completed == limit:
            checkpoint = save()
            last_saved = completed
        if completed % config['evaluation']['every'] == 0 or completed == horizon:
            if last_saved != completed:
                checkpoint = save()
                last_saved = completed
            draw()
        if stop():
            if last_saved != completed:
                checkpoint = save()
            LOG.info('Checkpointed stop at update %s', completed)
            return completed
    return completed


def prepare_run(config, args, output):
    """Freeze scientific configuration, source files, and packed-data metadata."""
    from utils.files import sha256, write_json
    path = output/'run.json'
    previous = json.loads(path.read_text()) if args.resume else None
    if previous is None and output.exists():
        raise FileExistsError(f'Output exists: {output}; use --resume or a new output')
    if previous is not None:
        config['initialization'] = previous['config']['initialization']
        source = previous['source']
    else:
        if args.init_checkpoint:
            config['initialization']['checkpoint'] = str(resolve(args.init_checkpoint))
        checkpoint = Path(config['initialization']['checkpoint'])
        source = dict(path=str(checkpoint), bytes=checkpoint.stat().st_size, sha256=sha256(checkpoint))
        if not args.init_checkpoint:
            key = 'priming_'+('b' if config['model']['model'] == 'OST_B' else 'l')+'_160'
            release = json.loads((ROOT/'old/checkpoints/pretraining_manifest.json').read_text())[key]
            if (source['bytes'], source['sha256']) != (release['checkpoint_bytes'], release['checkpoint_sha256']):
                raise ValueError('Default initialization differs from the pinned release; use --init-checkpoint for your own source')
    files = [ROOT/'posttrain.py', ROOT/'eval.py']
    for folder in ('models', 'losses', 'data', 'utils', 'metrics'):
        files.extend(sorted((ROOT/folder).rglob('*.py')))
    runtime = {str(file.relative_to(ROOT)): sha256(file) for file in files}
    packed = Path(config['data']['data_path'])/'imagenet-latents-images'
    metadata = {name: sha256(packed/name) for name in ('dataset_info.json', 'state.json')}
    shards = {file.name: dict(bytes=file.stat().st_size, mtime_ns=file.stat().st_mtime_ns)
              for file in sorted(packed.glob('*.arrow'))}
    if not shards:
        raise FileNotFoundError('No packed ImageNet Arrow shards')
    record = dict(version=1, config=config, source=source, runtime=runtime, data_metadata=metadata, arrow_files=shards)
    if previous is not None:
        import copy as _copy
        def _strip_manifest(rec):
            rec = _copy.deepcopy(rec)
            rec.pop('runtime', None)
            rec.pop('source', None)
            cfg = rec.get('config') or {}
            cfg.get('train', {}).pop('steps', None)
            cfg.get('train', {}).pop('save_every', None)
            cfg.pop('_provenance', None)
            cfg.pop('initialization', None)
            rec['config'] = cfg
            return rec
        if _strip_manifest(previous) != _strip_manifest(record):
            raise ValueError('Run configuration, source code, or packed-data metadata changed')
    if previous is None:
        write_json(path, record)
    config['_provenance'] = record['runtime'] | {'data_metadata': metadata, 'arrow_files': shards, 'source': source}
    return config


def train(config, args, output, hf_home, torch_home):
    import importlib
    import torch
    import torch.distributed as dist
    import torch._dynamo
    from data.imagenet import ClasswiseOTStream, RealImages
    # Preserve the original dataset-backend initialization before RNG seeding.
    importlib.import_module('datasets')
    from eval import evaluate
    from utils.distributed import initialize, broadcast_model, stopping
    from utils.posttrain_assets import evaluation_assets
    from utils.posttrain_checkpoint import initialize_posttraining, resume_posttraining, snapshot_rng, restore_rng
    from utils.random import seed_all
    rank, world = initialize('Post-training')
    reference = None
    try:
        torch.set_num_threads(int(os.environ['OMP_NUM_THREADS']))
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.optimize_ddp = False
        logging.basicConfig(level=logging.INFO if rank == 0 else logging.WARNING, format='%(asctime)s %(message)s')
        message = [None]
        if rank == 0:
            try:
                weights = {name: str(path) for name, path in evaluation_assets(config['evaluation'],
                    hf_home=hf_home, torch_home=torch_home).items()}
                message[0] = dict(config=prepare_run(config, args, output), weights=weights)
            except Exception as exc:
                message[0] = dict(error=repr(exc))
        dist.broadcast_object_list(message, src=0)
        if 'error' in message[0]:
            raise RuntimeError(message[0]['error'])
        config, weights = message[0]['config'], message[0]['weights']
        seed_all(config['train']['seed']+rank)
        model = make_model(config)
        matching = config['matching']
        matcher = make_matcher(config['data']['encoder_cache'], stack=matching['stack'],
            microbatch=matching['microbatch'], hf_home=hf_home, torch_home=torch_home,
            kind=matching['kind'], coefficient=matching['coefficient'], global_classes=matching['global_classes_per_step'],
            samples=matching['samples_per_class'], references=matching['real_references_per_class'],
            epsilon_mixture=matching['epsilon_mixture'], iterations=matching['num_iterations'], tolerance=matching['tolerance'])
        from losses.alignment import make_alignment
        alignment = make_alignment(config, model, ot_encoder=matcher.encoder)
        reference = RealImages(config['data']['encoder_cache'], config['data']['data_path'], workers=config['data']['reference_workers'])
        if matching['stack'] == 'convnext_full_mae':
            reference.validate_calibration(matcher.encoder.calibration)
        stream = ClasswiseOTStream(class_counts=reference.class_counts, rank=rank, world_size=world, device='cuda',
            seed=matching['seed'], global_classes_per_step=matching['global_classes_per_step'],
            samples_per_class=matching['samples_per_class'], real_references_per_class=matching['real_references_per_class'])
        checkpoint, completed = None, 0
        if args.resume:
            if args.resume == 'latest':
                latest = json.loads((output/'checkpoints/latest.json').read_text())
                checkpoint = output/'checkpoints'/latest['checkpoint']
            else:
                checkpoint = resolve(args.resume)
            optimizer, ema, completed = resume_posttraining(checkpoint, model, matcher, stream, config['optimizer'], config=config)
        else:
            optimizer, ema = initialize_posttraining(model, config['initialization']['checkpoint'], config['optimizer'], config=config)
        if alignment is not None:
            template = dict(optimizer.param_groups[0])
            template['params'] = list(alignment.proj.parameters())
            optimizer.add_param_group(template)
        recovered_rng = snapshot_rng() if args.resume else None
        broadcast_model(model)
        if config['train']['compile']:
            model.net.compile(fullgraph=True)
        if recovered_rng is not None:
            # Compilation setup belongs before recovery, as in the original driver.
            restore_rng(recovered_rng)
        def stop():
            local = (output/'STOP_REQUESTED').is_file()
            flag = torch.tensor(int(local), device='cuda')
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            return stopping('cuda') or bool(flag.item())
        return run_training(config, model, optimizer, ema, matcher, reference, stream, output,
            completed=completed, checkpoint=checkpoint, smoke_steps=args.smoke_steps, stop=stop,
            alignment=alignment,
            evaluate=lambda: evaluate(model, ema, config['evaluation'], config['sampling'], weights))
    finally:
        if reference is not None:
            reference.close()
        dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/posttrain/b160.yaml')
    parser.add_argument('--output')
    parser.add_argument('--resume', nargs='?', const='latest', metavar='CHECKPOINT')
    parser.add_argument('--init-checkpoint')
    parser.add_argument('--data-path')
    parser.add_argument('--encoder-cache')
    parser.add_argument('--hf-home')
    parser.add_argument('--torch-home')
    parser.add_argument('--smoke-steps', type=int)
    parser.add_argument('--check-config', action='store_true')
    args = parser.parse_args(argv)
    if args.resume and args.init_checkpoint:
        parser.error('--init-checkpoint starts a fresh run; --resume restores its saved state')
    if args.smoke_steps is not None and args.smoke_steps < 1:
        parser.error('--smoke-steps must be positive')
    config = load_config(args.config)
    for name in ('data_path', 'encoder_cache'):
        override = getattr(args, name)
        if override:
            config['data'][name] = str(resolve(override))
    if args.check_config:
        print(json.dumps(config, indent=2))
        return 0
    if args.output is None:
        parser.error('--output is required for training')
    output = resolve(args.output)
    hf_home, torch_home = configure_environment(output, phase='posttraining',
        hf_home=args.hf_home, torch_home=args.torch_home)
    train(config, args, output, hf_home, torch_home)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
