"""Trains the mini-VLM on COCO captions (train2017, ~118k images, one of the 5 captions drawn at each step).

  stage 1: connector only (encoder and language model frozen), as the first stage of LLaVA
  stage 2: connector + LoRA on the language model, starting from the stage 1 connector

The encoder stays frozen in both stages and is identical to the original one, so the universal perturbation
computed by uap_encoder.py does not depend on this training.
Saves a checkpoint every --save_every steps: relaunching the same command resumes the training.
"""
import argparse
import json
import math
import random
import time

import shutil

import torch

from common import (MODELS, UNFROZEN, base_encoder, unfrozen_encoder_path, ENCODERS, CocoCaptions, Connector, Encoder, MiniVLM, SEED, add_lora, amp_dtype, get_device, load_lm,
                    results_dir)

DEFAULTS = {1: dict(batch_size=64, lr=1e-3, epochs=1), 2: dict(batch_size=32, lr=2e-4, epochs=1)}


def lr_lambda(total, warmup=100):
    def f(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    return f


def lora_state(lm):
    from peft import get_peft_model_state_dict
    return get_peft_model_state_dict(lm)


def set_lora_state(lm, state):
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(lm, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mini-VLM training (connector, then connector + LoRA)')
    parser.add_argument('--encoder', required=True, choices=MODELS)
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2])
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--micro_batch_size', type=int, default=16,
                        help='images per forward pass; the gradient is accumulated over the batch (GPU memory)')
    parser.add_argument('--max_images', type=int, default=None, help='only the first N COCO images (quick test)')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=300)
    parser.add_argument('--enc_lr', type=float, default=2e-6,
                        help='<encoder>-unfrozen only: learning rate of the encoder in stage 2 (usual value for VLMs)')
    parser.add_argument('--train_blocks', type=int, default=-1,
                        help='<encoder>-unfrozen only: -1 = whole encoder; N = last N blocks (GPU memory)')
    args = parser.parse_args()
    for k, v in DEFAULTS[args.stage].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    torch.manual_seed(SEED)
    device = get_device()
    out = results_dir(args.encoder)
    unfrozen = args.encoder.endswith(UNFROZEN)
    if unfrozen and args.stage == 1:
        # Stage 1 is the same as for the frozen model (connector only): reuse it, so that both models only differ
        # by stage 2
        src = results_dir(base_encoder(args.encoder)) / 'connector_stage1.pt'
        assert src.exists(), '%s missing: train the frozen model first' % src
        if not (out / 'connector_stage1.pt').exists():
            shutil.copy(src, out / 'connector_stage1.pt')
        print('>> Stage 1 of %s = stage 1 of %s (copied)' % (args.encoder, base_encoder(args.encoder)))
        raise SystemExit(0)
    final = out / ('connector_stage%d.pt' % args.stage)
    ckpt_path = out / ('train_stage%d_ckpt.pt' % args.stage)
    if unfrozen:   # the checkpoint holds the encoder and its Adam state (~10 GB for a ViT-H): local disk
        unfrozen_encoder_path(args.encoder).parent.mkdir(parents=True, exist_ok=True)
        ckpt_path = unfrozen_encoder_path(args.encoder).parent / ('%s_train_stage2_ckpt.pt' % args.encoder)
    log_path = out / ('train_stage%d_log.jsonl' % args.stage)
    if final.exists():
        print('>> Already trained:', final)
        raise SystemExit(0)

    print('>> Loading', args.encoder, 'and the language model')
    encoder = Encoder(base_encoder(args.encoder)).to(device)   # the original weights (trained below if unfrozen)
    lm, tok = load_lm()
    connector = Connector(encoder.dim, lm.config.hidden_size)
    if args.stage == 1:
        for p in lm.parameters():
            p.requires_grad_(False)
    else:
        connector.load_state_dict(torch.load(out / 'connector_stage1.pt', map_location='cpu'))
        lm = add_lora(lm)
        lm.print_trainable_parameters()
    vlm = MiniVLM(encoder, connector, lm, tok).to(device)
    vlm.lm.train()
    vlm.connector.train()

    params = [p for p in list(connector.parameters()) + list(lm.parameters()) if p.requires_grad]
    enc_params = []
    if unfrozen:
        from defense_finetune import set_trainable
        enc_params = set_trainable(encoder, args.train_blocks)
        encoder.train()
        encoder.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    print('>> Trainable parameters: %.1fM (+ encoder %.1fM)' % (
        sum(p.numel() for p in params) / 1e6, sum(p.numel() for p in enc_params) / 1e6))
    data = CocoCaptions('train2017', max_images=args.max_images)
    steps_per_epoch = len(data) // args.batch_size
    total = steps_per_epoch * args.epochs
    groups = [{'params': params, 'lr': args.lr}] + ([{'params': enc_params, 'lr': args.enc_lr}] if enc_params else [])
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.0, foreach=not enc_params)
    params = params + enc_params
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(total))
    dtype = amp_dtype()
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16 and device.type == 'cuda'))
    print('>> %d images, %d steps per epoch, %d epochs, batch %d (micro-batches of %d), lr %g, autocast %s' % (
        len(data), steps_per_epoch, args.epochs, args.batch_size, args.micro_batch_size, args.lr, dtype))

    epoch, step = 0, 0
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location='cpu')
        connector.load_state_dict(ck['connector'])
        if unfrozen:
            encoder.load_state_dict(ck['encoder'])
        if args.stage == 2:
            set_lora_state(lm, ck['lora'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        scaler.load_state_dict(ck['scaler'])
        epoch, step = ck['epoch'], ck['step']
        print('>> Resuming from', ckpt_path, ': epoch', epoch, 'step', step)

    def save_checkpoint():
        ck = {'connector': connector.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
              'scaler': scaler.state_dict(), 'epoch': epoch, 'step': step}
        if args.stage == 2:
            ck['lora'] = lora_state(lm)
        if unfrozen:
            ck['encoder'] = encoder.state_dict()
        tmp = ckpt_path.parent / (ckpt_path.name + '.tmp')
        torch.save(ck, tmp)
        tmp.replace(ckpt_path)

    rng = random.Random(SEED)
    while epoch < args.epochs:
        # Same order of images after a resumption: the order only depends on the epoch
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(SEED + epoch)).tolist()
        order = order[step * args.batch_size:steps_per_epoch * args.batch_size]
        loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=order,
                                             num_workers=args.workers, pin_memory=True, drop_last=True)
        t0, losses = time.time(), []
        for x_u8, idx in loader:
            captions = [rng.choice(data.items[i][1]) for i in idx.tolist()]
            x = x_u8.to(device, non_blocking=True).float() / 255
            opt.zero_grad(set_to_none=True)
            mb, loss = args.micro_batch_size, 0.0
            for c in range(0, len(x), mb):  # gradient accumulation: same batch, less GPU memory
                with torch.autocast(device.type, dtype=dtype, enabled=device.type == 'cuda'):
                    with torch.set_grad_enabled(unfrozen):
                        tokens = encoder(x[c:c + mb])
                    part = vlm.caption_loss(tokens.float(), captions[c:c + mb]) * len(tokens) / len(x)
                scaler.scale(part).backward()
                loss += part.item()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            losses.append(loss)
            if step % 50 == 0:
                rec = {'stage': args.stage, 'epoch': epoch, 'step': step, 'of': steps_per_epoch,
                       'loss': sum(losses[-50:]) / len(losses[-50:]), 'lr': sched.get_last_lr()[0],
                       'img_per_s': 50 * args.batch_size / (time.time() - t0)}
                t0 = time.time()
                print('epoch %d step %d/%d  loss %.3f  lr %.2e  %.0f img/s' % (
                    epoch, step, steps_per_epoch, rec['loss'], rec['lr'], rec['img_per_s']))
                with open(log_path, 'a') as f:
                    f.write(json.dumps(rec) + '\n')
            if step % args.save_every == 0:
                save_checkpoint()
        epoch, step = epoch + 1, 0
        save_checkpoint()

    # Final weights; the checkpoint is no longer needed
    if unfrozen:   # before the connector, so that an existing connector_stage2.pt always has its encoder
        encoder.net.gradient_checkpointing_disable()
        for p in encoder.parameters():
            p.requires_grad_(False)
        torch.save(encoder.state_dict(), unfrozen_encoder_path(args.encoder))
        print('>> Saved the post-trained encoder to', unfrozen_encoder_path(args.encoder))
    torch.save(connector.state_dict(), final)
    if args.stage == 2:
        lm.save_pretrained(str(out / 'lora_stage2'))
    ckpt_path.unlink(missing_ok=True)

    # A few captions of COCO val images, to check by eye that the model describes images
    vlm.eval()
    val = CocoCaptions('val2017', max_images=5)
    x = torch.stack([val[i][0] for i in range(len(val))]).to(device).float() / 255
    for (path, refs), cap in zip(val.items, vlm.caption(x)):
        print('  model    :', cap)
        print('  reference:', refs[0])
    print('>> Saved', final)
