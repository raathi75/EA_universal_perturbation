# Mini-VLMs: four pre-trainings against a universal perturbation

Does the way a vision encoder was pre-trained change its vulnerability to a universal perturbation?
Four mini-VLMs are built with **exactly the same recipe**; only the frozen encoder differs:

| Name | Model | Pre-training | Size |
|---|---|---|---|
| `siglip` | `google/siglip-so400m-patch14-224` | contrastive image–text (sigmoid), WebLI | ~400M |
| `clip` | `openai/clip-vit-large-patch14` | contrastive image–text (softmax), 400M pairs | ~300M |
| `ijepa` | `facebook/ijepa_vith14_1k` | predicts hidden parts **in feature space**, ImageNet-1k, no text | ~630M |
| `mae` | `facebook/vit-mae-huge` | reconstructs hidden **pixels**, ImageNet-1k, no text | ~630M |

All are ViTs with patch 14 at 224 px, giving 256 patch tokens (the CLS token of CLIP and MAE is dropped).
I-JEPA and MAE are both ViT-H/14 trained on ImageNet-1k: the cleanest pair, they differ only by the objective.
MAE runs without masking, and its patches are kept in their original order (ViTMAE shuffles them otherwise).

Common parts: connector (LayerNorm, 2 × 2 pixel shuffle → 64 tokens, 2-layer MLP, as in SmolVLM), language model
SmolLM2-135M-Instruct (the one of SmolVLM-256M), COCO train2017 captions, prompt "Describe this image in one
sentence.". The original pre-trained encoders are used, frozen in every stage.

**Attack**: same as the SmolVLM notebooks (label-free cosine loss on the tokens, Adam lr 1/255, batch 8,
10 passes, ξ = 10/255), built on the same 500 ImageNetV2 images (sorted files,
`np.random.RandomState(0).permutation`). Perturbed images are rounded to 8 bits.

**Evaluation**, two sets:
1. **COCO val2017 (1,000 images, never used for training), against the 5 human captions**: CIDEr; *wrong* = the
   caption is far from every human caption (best MiniLM similarity < 0.5), which measures whether captions become
   wrong and not merely different; *changed* = similarity with the clean caption < 0.5. The clean CIDEr is also the
   quality check: compare encoders only if it is close.
2. **ImageNetV2 held-out images of the SmolVLM report** (500, results also on the first 200 as in the report):
   captions changed, at 0.5 and 0.7.

Every rate has a 95% Wilson interval and is compared with random noise of the same budget. Each model is also
attacked with the perturbations of the three other encoders and of SmolVLM-256M (transfer).

## Files

| File | Role |
|---|---|
| [common.py](common.py) | Paths, split, encoders, connector, mini-VLM, metrics |
| [prepare_data.sh](prepare_data.sh) | Downloads ImageNetV2 and COCO 2017 to the local disk (~21 GB) |
| [uap_encoder.py](uap_encoder.py) | Universal perturbation on the encoder alone, token damage (+ transfer) |
| [train_connector.py](train_connector.py) | Stage 1: connector; stage 2: connector + LoRA on the LM (resumable) |
| [eval_coco.py](eval_coco.py) | COCO val: quality and attack against the human captions |
| [evaluate_captions.py](evaluate_captions.py) | ImageNetV2 held-out images: captions changed |
| [compare.py](compare.py) | Tables of the four encoders → `results/comparison.md` |
| [run_minivlm.sh](run_minivlm.sh) | Everything for one or several encoders, resumable |
| [defense_finetune.py](defense_finetune.py) | Defense: fine-tuning of the encoder (FT-fixed, FT-UAT) + adaptive attack |
| [run_defense.sh](run_defense.sh) | The defense for one or several encoders, once their mini-VLM is trained |
| [smolvlm_reference.py](smolvlm_reference.py) | SmolVLM-256M on the same images and metrics; was its encoder changed by HuggingFace? |
| [run_converged.sh](run_converged.sh) | Every attack again, to convergence, with the connector-output attack and the defense |

## Running on two school machines

```bash
pip install -r requirements.txt          # once, in the venv (home directory, shared)
./run_minivlm.sh ijepa clip              # machine 1, in tmux
./run_minivlm.sh mae siglip              # machine 2, in tmux
# once both are done: the same two commands again (they only add the missing transfers), then
python compare.py
```

Data and model weights go to `/Data/gurvan.nkouathio-kemnhou/minivlm` (local disk of each machine, `MINIVLM_DATA`);
results to `results/<encoder>/` (home directory, shared). Every step can be relaunched: finished steps are
skipped, the training resumes from its checkpoint (saved every 300 steps), evaluations only add what is missing.

## Defense: fine-tuning the encoder (as in the report, notebook 07)

Once an encoder's mini-VLM is trained: `./run_defense.sh ijepa clip` (machine 1), `./run_defense.sh mae siglip`
(machine 2). The whole encoder is fine-tuned on the 500 ImageNetV2 build images (3 epochs, Adam lr 1e-5, batch 8),
with the anchoring loss of the report; the connector and the language model are not retrained. Two variants:
**FT-fixed** (against the fixed universal v) and **FT-UAT** (v updated at each step, Shafahi et al. 2020). Each
fine-tuned encoder is then attacked by an **adaptive** v (warm start from the original v, 7 epochs). Results in
`results/<encoder>/defense_ft.json`: clean cost (vs the original mini-VLM), fixed v, adaptive v, on COCO val
(CIDEr, wrong, changed) and on the ImageNetV2 held-out images (changed). Fine-tuned encoders (~2.5 GB for a ViT-H)
stay on the local disk: run each encoder on one machine. Out of GPU memory: `TRAIN_BLOCKS=8 ./run_defense.sh ...`.

## Attacks run to convergence (run_converged.sh)

Every universal perturbation (token v, connector-output v, adaptive v of the defense) is optimised until it stops
improving (`common.converge_uap`): after each epoch, v is scored on the 500 build images (rounded to 8 bits); the
best v is kept; on a plateau (no gain of 0.001 for 2 epochs) the optimisation restarts from the best v with half the
step; it stops after 3 halvings, never before the epochs of the report (10, and 7 for the adaptive attack), at most
60 epochs. Only build images are used to choose v; captions changed are measured on the held-out images (COCO val
and the 500 ImageNetV2 held-out images). The **connector-output v** is built on the 64 vectors the language model
receives instead of the 256 encoder tokens. `./run_converged.sh ijepa clip` (espagne), `./run_converged.sh mae siglip`
(allemagne), then the same again for the transfers, then `python compare.py`. The 10-epoch results are kept in
`results/old_10epochs/`.
