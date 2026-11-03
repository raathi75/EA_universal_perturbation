# Universal Adversarial Perturbations: replication and beyond

This repository holds our work for the EA Adversarial ML course, built around the paper **"Universal Adversarial Perturbations"** by Moosavi-Dezfooli, Fawzi, Fawzi and Frossard (CVPR 2017). Our goal is simple to state: first, rebuild the paper's main experiment ourselves and check whether we get the same results; then, look at where the idea has gone since 2017, especially with newer models such as Vision Transformers.

## The idea in a few words

Image classifiers can be fooled by tiny changes to an image that a person would barely notice. Usually, though, each image needs its own carefully computed change. The surprising finding of this paper is that **a single, fixed noise pattern** can be added to almost *any* image and still change what the network predicts most of the time. In other words, one "universal" pattern fools the model across many different photos.

To build that pattern, the paper goes through a set of images one by one. Whenever the current pattern does not fool an image yet, it computes the smallest nudge that would push that image over the model's decision line (a method called DeepFool), adds the nudge to the pattern, and then shrinks the pattern back so it stays small enough to be almost invisible. After a few passes over the images, the pattern fools most of them.

## Project goals and progress

| Goal | Status |
|---|---|
| Rebuild the paper's algorithm in PyTorch and run it on Google Colab | ✅ Done |
| Reproduce the paper's main experiments (tables and figures) | ✅ Done |
| Explain the differences between our numbers and the paper's | ✅ Done |
| Find recent papers (2024–2026) that build on this work | ✅ Done |
| Test the original algorithm on a Vision Transformer (ViT) | 🟡 Code ready, results pending |

## What is in this repository

```
EA_universal_perturbation/
├── UAP Notebooks/
│   ├── universal_adversarial_perturbations_replication.ipynb         ← main replication (with outputs)
│   ├── universal_adversarial_perturbations_replication_E7_vit.ipynb  ← same notebook + summary + ViT experiment
│   └── CNN Experiment utils/
│       ├── universal_googlenet_linf.npy    ← the universal pattern we computed for GoogLeNet
│       ├── v_googlenet_linf_ckpt.pt        ← checkpoint of the same pattern
│       └── results_googlenet_linf.json     ← all numbers from our main run
└── Interesting Papers/
    ├── uap_citing_papers.pdf               ← our summary of three recent papers (with diagrams)
    ├── 2502.21048v2.pdf                    ← PSP-UAP (CVPR 2025)
    ├── 2406.05491v4.pdf                    ← C-PGC (ICCV 2025)
    └── 2025.naacl-long.302.pdf             ← IRIS (NAACL 2025)
```

## Part 1: the replication

### How we set it up

We followed the authors' original code closely, but we had to adapt a few things to make it run on a free Colab GPU. Instead of the original ImageNet images, which require registration, we used **ImageNetV2**, a freely available set of 10,000 photos covering the same 1,000 categories. We used 4,000 of these images to build the pattern and kept the other 6,000 aside to test it on photos it had never seen. The main model is **GoogLeNet**, and the pattern is limited to a change of at most 10 out of 255 brightness levels per pixel, which is the same limit used in the paper.

### What we found

The notebook reproduces six experiments from the paper, named E1 to E6.

| Experiment | What it checks | Our result | Paper |
|---|---|---|---|
| **E1** (Table 1) | How often the pattern fools unseen images | **54.6 %** | 78.9 % |
| **E2** (Fig. 8) | Universal pattern vs. random noise of the same size | Universal fools **2 to 3 times more** images | Same conclusion |
| **E3** (Fig. 6) | Does using more images to build the pattern help? | Yes: 15.6 % → 22.9 % → 36.3 % → 54.6 % with 500 → 4,000 images | Same trend |
| **E4** (Table 2) | Does the GoogLeNet pattern also fool other models? | VGG-16: 51.6 %, ResNet-152: 34.9 % | 39.2 %, 45.5 % |
| **E5** (Figs. 1, 3, 4) | What the pattern and the fooled images look like | Visualisations in the notebook | — |
| **E6** (Fig. 7) | Which wrong labels the pattern pushes images toward | Mostly "brain coral" (311 images), then "paper towel" and "tarantula" | Same "dominant label" effect |

Overall, the main claims of the paper hold up. A single pattern fools more than half of the unseen images, it works far better than random noise, and it even carries over to other networks.

### Why our main number is lower than the paper's

Our 54.6 % is below the paper's 78.9 %, and there are clear reasons for that. First, we built the pattern from 4,000 images instead of 10,000. The paper's own Figure 6 shows that with 4,000 images GoogLeNet reaches about 69 %, so roughly 10 points of the gap come from this choice alone. Second, our run stopped improving after the fourth pass, at about 59 % on the training images. Because the pattern is capped at a small size, many pixels hit that limit, and from then on most new nudges get cut off. Finally, we used ImageNetV2 and PyTorch's pretrained models rather than the original data and models, which also shifts the numbers somewhat.

## Part 2: where the idea went next

Following our professor's advice, we used Google Scholar's "Cited by" feature to find recent, peer-reviewed papers that build on this work. We picked three that show how far the idea has travelled:

- **PSP-UAP (CVPR 2025)** builds universal patterns *without any real images*. Interestingly, it starts from the same "dominant label" effect we saw in E6: the pattern itself looks like an image to the network, so pieces of it can be used as training data. The authors also admit their method works poorly on Vision Transformers.
- **C-PGC (ICCV 2025)** brings the idea to models that understand both images and text, such as CLIP. They first tried the original algorithm on these models and found it did not work well, which led them to design a new approach that attacks the link between images and their captions.
- **IRIS (NAACL 2025)** applies the same "one pattern fools many inputs" idea to large language models, using a single text suffix that works across many different requests.

Our summary in `Interesting Papers/uap_citing_papers.pdf` analyses all three with the same six questions, includes diagrams of each method, and ends with the common thread: **one perturbation works on many inputs when it attacks something those inputs share.**

## Part 3: Vision Transformers (in progress)

The literature review pointed to an open question. Earlier work compared Vision Transformers and CNNs under universal attacks, but it used a different algorithm, not the paper's original one. So we added experiment **E7** to the second notebook to answer two things. First, does the pattern we built for GoogLeNet also fool a ViT (ViT-B/16)? Second, if we run the original algorithm directly on the ViT, how far does it get compared with GoogLeNet under the same settings? The code is written and ready to run; the results will be added here once the run is finished.

## How to run the notebooks

1. Open either notebook in **Google Colab** and switch the runtime to a GPU (`Runtime → Change runtime type → T4 GPU`).
2. Run the cells from the top. The notebook downloads ImageNetV2 automatically (about 1.3 GB) and saves its results to a folder called `uap_replication` in your Google Drive.
3. The full main run takes around 20 minutes on a T4 GPU, and the extra experiments add some more time.

**For the ViT experiment only**, you can skip the long main run. Copy the three files from `CNN Experiment utils/` into `MyDrive/uap_replication/`, run cells 2 to 13 of the E7 notebook as usual, skip the main computation cell, and then run the E7 cells at the end. They load our saved pattern instead of recomputing it.

## Differences from the original paper

To keep things transparent, here is everything we changed:

- PyTorch and torchvision pretrained models instead of the original Caffe models.
- ImageNetV2 instead of the original ImageNet validation set.
- 4,000 images to build the pattern instead of 10,000, to fit within Colab's time limits.
- DeepFool only looks at the 10 most likely classes, just as in the authors' official code.

## Next steps

- Run E7 and add the ViT results to this README.
- Bring together the work from the whole team, including the ViT replication and the theoretical part, into one shared story.
- If time allows, run the main experiment with all 10,000 images to see how close we get to the paper's 78.9 %.

## References

1. S.-M. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, P. Frossard. *Universal adversarial perturbations*. CVPR 2017. Official code: [LTS4/universal](https://github.com/LTS4/universal)
2. C. Lee, Y. Song, J. Son. *Data-free Universal Adversarial Perturbation with Pseudo-semantic Prior*. CVPR 2025. [arXiv:2502.21048](https://arxiv.org/abs/2502.21048)
3. H. Fang et al. *One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models*. ICCV 2025. [arXiv:2406.05491](https://arxiv.org/abs/2406.05491)
4. D. Huang et al. *Stronger Universal and Transferable Attacks by Suppressing Refusals*. NAACL 2025. [ACL Anthology](https://aclanthology.org/2025.naacl-long.302/)
5. P. Benz et al. *Adversarial Robustness Comparison of Vision Transformer and MLP-Mixer to CNNs*. 2021. [arXiv:2110.02797](https://arxiv.org/abs/2110.02797)
6. ImageNetV2 dataset: [Hugging Face](https://huggingface.co/datasets/vaishaal/ImageNetV2)