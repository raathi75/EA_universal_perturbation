# Universal Adversarial Perturbations: replication and beyond

This repository holds our team's work for the EA Adversarial ML course, built around the paper **"Universal Adversarial Perturbations"** by Moosavi-Dezfooli, Fawzi, Fawzi and Frossard (CVPR 2017). We started with a simple question: if we rebuild the paper's main experiment ourselves, do we get the same results? Once that was settled, we moved on to three follow-up questions. Does the attack still work on newer models such as Vision Transformers? Can a simple defense stop it? And where has the idea gone in the research since 2017?

## The idea in a few words

Image classifiers can be fooled by tiny changes to an image that a person would barely notice. Usually, though, each image needs its own carefully computed change. The surprising finding of this paper is that **a single, fixed noise pattern** can be added to almost *any* image and still change what the network predicts most of the time. In other words, one "universal" pattern fools the model across many different photos.

To build that pattern, the paper goes through a set of images one by one. Whenever the current pattern does not fool an image yet, it computes the smallest nudge that would push that image over the model's decision line (a method called DeepFool), adds the nudge to the pattern, and then shrinks the pattern back so it stays almost invisible. After a few passes over the images, the pattern fools most of them.

![The same pattern added to six different photos: five of them are now recognised as "spider web"](Experiments%20ViT/python/results/imagenet_resnet50_10k/examples_resnet50.png)

*Top row: original photos. Bottom row: the same photos with our universal pattern for ResNet-50 added. Labels in red were fooled, and almost all of them become "spider web".*

## Project goals and progress

| Goal | Status |
|---|---|
| Rebuild the paper's algorithm in PyTorch | ✅ Done, in two independent versions |
| Check that DeepFool, the building block, works as expected | ✅ Done |
| Reproduce the paper's main result on ImageNet | ✅ Done: 81.0 % on ResNet-50 (paper: 78–84 %) |
| Reproduce the paper's other experiments (random noise, transfer, dominant labels…) | ✅ Done |
| Test the original algorithm on modern models, including Vision Transformers | ✅ Done, on six architectures |
| Test a simple defense (adding random noise to the input) | ✅ Done |
| Test that defense against an attacker who knows about it | ✅ Done |
| Find recent papers that build on this work | ✅ Done |
| Check whether a pattern built for GoogLeNet also fools a ViT | 🟡 Code ready, results pending |

## What is in this repository

The work is split into two main folders, plus a folder of reading material.

```
EA_universal_perturbation/
├── UAP Notebooks/          ← Colab replication (GoogLeNet, ImageNetV2)
│   ├── universal_adversarial_perturbations_replication.ipynb
│   ├── universal_adversarial_perturbations_replication_E7_vit.ipynb
│   └── CNN Experiment utils/      ← saved pattern and results of the Colab run
│
├── Experiments ViT/        ← full PyTorch port of the authors' code (ImageNet)
│   ├── python/                    ← DeepFool, the algorithm, evaluation and defense scripts
│   │   ├── results/               ← every pattern, number and figure from these experiments
│   │   └── testlogistic/          ← sanity checks of DeepFool
│   ├── docs/en/                   ← project overview and analyses (PDF, English)
│   ├── docs/                      ← same documents in French
│   ├── reports/, research_notes/  ← literature review on defenses (French)
│   └── matlab/, precomputed/      ← the authors' original MATLAB code and patterns, unchanged
│
└── Interesting Papers/     ← our summary of three recent papers that cite the original
```

## Part 1: reproducing the paper

### Two replications, two settings

We reproduced the paper twice, and the two versions complement each other nicely.

The **Colab notebook** (`UAP Notebooks/`) is a light version designed to run on a free GPU. Instead of the original ImageNet images, which require registration, it uses **ImageNetV2**, a freely available set of 10,000 photos covering the same 1,000 categories. It builds the pattern for **GoogLeNet** from 4,000 images and tests it on the other 6,000.

The **PyTorch port** (`Experiments ViT/python/`) is a rewrite of the authors' own code. Because it ran on the school's GPU machines, it could follow the paper's protocol much more closely: it builds the pattern from 10,000 ImageNet images (10 per category) and tests it on 40,000 other images the algorithm never saw. In both cases, the pattern is limited to a change of at most 10 out of 255 brightness levels per pixel, exactly as in the paper.

### Checking DeepFool first

Before trusting the full algorithm, we checked its building block. On a simple model (a logistic regression on 2D points), the smallest change that flips a prediction can be computed exactly by hand, so DeepFool should find it in a single step. It did: all 803 points were fooled, always in one step, and the result matched the exact answer to within 0.000003. On a real network (ResNet-34), DeepFool turned a "macaw" into a "flamingo" by changing the image by only 0.5 %.

### Results

| | Colab notebook | PyTorch port | Paper |
|---|---|---|---|
| Model | GoogLeNet | ResNet-50 | GoogLeNet / ResNet-152 |
| Images used to build the pattern | 4,000 (ImageNetV2) | 10,000 (ImageNet) | 10,000 (ImageNet) |
| **Images fooled (unseen photos)** | **54.6 %** | **81.0 %** | 78.9 % / 84.0 % |
| Random noise of the same size | fools 2–3× fewer images | fools 9.5 % | about 10 % |

Overall, the paper's main claim holds up very clearly. With the paper's full protocol, a single pattern fools **81.0 %** of 40,000 unseen photos on ResNet-50, and the model's accuracy drops from 80.5 % to 17.1 %. Meanwhile, a control pattern with exactly the same pixel values but shuffled into a random order fools only 9.5 %, which shows that it is the *structure* of the pattern that matters, not just its size.

The Colab notebook reproduces the paper's other experiments as well, and they all agree with the original: more images give a stronger pattern (15.6 % with 500 images, 54.6 % with 4,000), the pattern carries over to other networks (51.6 % on VGG-16, 34.9 % on ResNet-152), and it pushes most images toward a few "dominant" wrong labels.

### Why the Colab number is lower

The Colab result (54.6 %) sits below the paper's 78.9 %, and the PyTorch port helps explain why. The main reason is the number of images: the paper's own Figure 6 shows that GoogLeNet reaches only about 69 % with 4,000 images, so a good part of the gap comes from that choice alone. In addition, the Colab run stopped improving after its fourth pass, because many pixels had already hit the size limit. Once the PyTorch port used the full 10,000 images from the original dataset, the result landed right inside the paper's range. In other words, the gap came from our lighter setup, not from a problem with the method.

A small test on 500 images (ResNet-18, Imagenette) tells the same story from the other side. There, the pattern fooled 85.6 % of the images it was built on but only 24.0 % of new ones: with so few examples, it simply memorises them instead of finding a truly universal direction.

## Part 2: does it still work on modern models?

Next, we ran the same algorithm, with the same settings, on six pretrained models from three families: classic CNNs (ResNet-18, ResNet-50), a modern CNN (ConvNeXt-B), and three models built on attention (MaxViT-T, Swin-B and a plain Vision Transformer, ViT-B/16).

| Model | Family | Accuracy before | Accuracy after | **Images fooled** | Pushed toward |
|---|---|---|---|---|---|
| ResNet-18 | CNN | 68.8 % | 18.0 % | **79.6 %** | brain coral (18 %) |
| ResNet-50 | CNN | 80.5 % | 17.1 % | **81.0 %** | spider web (97.5 %) |
| ConvNeXt-B | modern CNN | 83.9 % | 15.5 % | **82.9 %** | brain coral (97.6 %) |
| MaxViT-T | CNN + attention | 83.5 % | 23.0 % | **74.9 %** | brain coral (98.9 %) |
| Swin-B | Transformer | 83.3 % | 23.9 % | **73.8 %** | brain coral (97.8 %) |
| ViT-B/16 | Transformer | 80.8 % | 25.7 % | **71.9 %** | scoreboard (96.8 %) |

*"Pushed toward" is the share of fooled images that end up with that label.*

Three things stand out.

**First, the attack works on every model**, including Vision Transformers. Even in the worst case, the pattern fools more than 7 images out of 10, while the shuffled control never goes above 18 %. This directly answers one of our starting questions: the paper's original algorithm does carry over to Transformers.

**Second, attention-based models resist somewhat better.** The CNNs are fooled 80–83 % of the time, against 72–75 % for the attention-based models. The algorithm also needs more passes on them, and on ViT it never reached its 80 % target, levelling off at about 73 %. Interestingly, ConvNeXt is the most accurate model and yet the easiest to fool, so being more accurate does not mean being more robust. This matches what recent papers report, since PSP-UAP (CVPR 2025) also found that universal patterns work less well on ViTs.

**Third, the dominant label is surprisingly consistent.** Four very different models, computed independently, all push images toward the same label: "brain coral". Our Colab GoogLeNet run points the same way, since its most common wrong label is also brain coral. A likely explanation is that brain coral, like spider web, is a category defined almost entirely by a texture, which a repeated pattern can imitate easily. ViT is the exception: its pattern is visibly a grid of 16×16-pixel squares, exactly matching how ViT cuts images into patches, and it sends images to "scoreboard", a category made of rectangular cells. Indeed, the ResNet-50 pattern added to a plain gray image is classified as "spider web" with 99 % confidence, so the pattern really is a spider-web texture as far as the network is concerned.

## Part 3: can we defend against it?

### A simple defense that seems to work

Because DeepFool pushes each image *just barely* over the decision line, the pattern should be fragile. To test this, we added a little random noise to every image before classifying it. The result looked very promising at first: with a moderate amount of noise, the fooling rate dropped from about 80 % to between 7 % and 21 % depending on the model, while normal accuracy fell by only 4–9 points on most models. ResNet-18 was the exception, losing 17 points to the noise alone.

### …until the attacker knows about it

However, a defense only counts if it holds up against an attacker who knows it is there. So we recomputed the pattern with a standard technique (Expectation Over Transformation, or EOT), which simply means building the pattern on many noisy copies of each image so that it learns to survive the noise.

| Pattern (ResNet-50, defense noise σ = 20/255) | Fooled without defense | Fooled with defense |
|---|---|---|
| Paper's algorithm, unaware of the defense | 80.1 % | 14.0 % |
| Gradient-based, unaware of the defense | 83.4 % | 15.8 % |
| Gradient-based, **aware of the defense**, same size limit | 50.8 % | **44.6 %** |
| Gradient-based, **aware of the defense**, size limit 16/255 | 93.3 % | **85.5 %** |

As expected from the literature, the defense breaks. At the same size limit, the aware attacker still fools 45–50 % of the images through the noise, and with a slightly larger pattern it reaches 86 %. In other words, the noise does not stop the attack; it mostly forces the attacker to use a bigger pattern. It also changes what the pattern looks like: the noise-resistant patterns are coarser, with larger stripes, and they no longer push images toward spider web.

## Part 4: what the research says

Alongside the experiments, we looked at the literature in two directions.

First, following our professor's advice, we used Google Scholar's "Cited by" feature to find recent, peer-reviewed papers that build on the original work. Our summary in `Interesting Papers/uap_citing_papers.pdf` analyses three of them with the same six questions and includes diagrams of each method:

- **PSP-UAP (CVPR 2025)** builds universal patterns *without any real images*, starting from the same "dominant label" effect we observed. Its authors also note that the method works poorly on ViTs.
- **C-PGC (ICCV 2025)** brings the idea to models that understand both images and text, such as CLIP. The original algorithm failed there, so they designed a new attack on the link between images and their captions.
- **IRIS (NAACL 2025)** applies the same "one pattern fools many inputs" idea to large language models, using a single text suffix.

Their common thread is a useful way to sum up the whole project: **one perturbation works on many inputs when it attacks something those inputs share.**

Second, a literature review on defenses (`Experiments ViT/docs/` and `reports/`) explains why our noise defense was expected to fall. Random input changes looked spectacular when they were first proposed in 2017–2018, but they were all broken by attackers who took the randomness into account. Our ImageNet results confirm this on universal patterns. As far as the review could find, no earlier study had measured both the defender's loss of accuracy and how the dominant label changes under this kind of attack.

## How to run the code

### Colab notebooks (`UAP Notebooks/`)

1. Open either notebook in **Google Colab** and switch the runtime to a GPU (`Runtime → Change runtime type → T4 GPU`).
2. Run the cells from the top. The notebook downloads ImageNetV2 automatically (about 1.3 GB) and saves its results to a folder called `uap_replication` in your Google Drive.
3. The main run takes around 20 minutes on a T4 GPU.

For the GoogLeNet-to-ViT experiment (E7), you can skip the long main run. Copy the three files from `CNN Experiment utils/` into `MyDrive/uap_replication/`, run cells 2 to 13 of the E7 notebook, skip the main computation cell, and then run the E7 cells at the end.

### PyTorch scripts (`Experiments ViT/python/`)

```bash
cd "Experiments ViT/python"
pip install -r requirements.txt

# Check DeepFool on the simple model
python testlogistic/testlogistic.py

# Build a pattern on a small image set, then evaluate it on new images
python demo.py -m resnet18 -t <image folder, one sub-folder per class> --num_imgs_per_class 50
python evaluate.py -m resnet18 -v <validation folder>
```

The code runs on an NVIDIA GPU, an Apple Silicon Mac or a plain CPU. To follow the paper's full ImageNet protocol, you need a free Hugging Face account to download the ImageNet validation set; after that, `run_models.sh` runs the six-model experiment and `run_eot_imagenet.sh` runs the defense experiment. Step-by-step instructions are in [`Experiments ViT/python/README.md`](Experiments%20ViT/python/README.md).

## Differences from the original paper

To keep things transparent, here is what we changed:

- PyTorch and torchvision pretrained models instead of the original Caffe models.
- **Colab notebook:** ImageNetV2 instead of ImageNet, and 4,000 images to build the pattern instead of 10,000, to fit within Colab's limits.
- **PyTorch port:** the 10,000 images used to build the pattern come from the ImageNet *validation* set (the paper used the training set), kept fully separate from the 40,000 evaluation images.
- DeepFool only looks at the 10 most likely classes, just as in the authors' official code.

## Limitations

A few caveats apply to all of these results. Each model was run only once, so differences of one or two points are not meaningful. We tested one model per family, which means we cannot fully separate the effect of the architecture from the way each model was trained. Moreover, the algorithm stops as soon as it fools 80 % of its images, so the models were compared at a fixed target rather than at a fixed amount of effort. Finally, the defense experiments were run on ResNet-50 only, with one noise level.

## Next steps

- Run E7 to see whether the GoogLeNet pattern also fools a ViT, and build a full **transfer table** between the six models (does Swin's "brain coral" pattern also fool ConvNeXt?).
- Repeat the defense experiment on a Transformer, since Transformers were easier to defend against the unaware attacker.
- Measure the full trade-off between the defender's accuracy and the attacker's pattern size.
- Study why the dominant label changes when the pattern has to survive noise.
- Repeat the main runs with several random seeds, and translate the remaining French documents.

## Credits and references

The PyTorch port is a fork of the authors' official code, [LTS4/universal](https://github.com/LTS4/universal); the `matlab/` and `precomputed/` folders are their original files, left unchanged.

1. S.-M. Moosavi-Dezfooli, A. Fawzi, O. Fawzi, P. Frossard. *Universal adversarial perturbations*. CVPR 2017. [arXiv:1610.08401](https://arxiv.org/abs/1610.08401)
2. S.-M. Moosavi-Dezfooli, A. Fawzi, P. Frossard. *DeepFool: a simple and accurate method to fool deep neural networks*. CVPR 2016. [arXiv:1511.04599](https://arxiv.org/abs/1511.04599)
3. C. Lee, Y. Song, J. Son. *Data-free Universal Adversarial Perturbation with Pseudo-semantic Prior*. CVPR 2025. [arXiv:2502.21048](https://arxiv.org/abs/2502.21048)
4. H. Fang et al. *One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models*. ICCV 2025. [arXiv:2406.05491](https://arxiv.org/abs/2406.05491)
5. D. Huang et al. *Stronger Universal and Transferable Attacks by Suppressing Refusals*. NAACL 2025. [ACL Anthology](https://aclanthology.org/2025.naacl-long.302/)
6. A. Athalye, N. Carlini, D. Wagner. *Obfuscated Gradients Give a False Sense of Security*. ICML 2018. [arXiv:1802.00420](https://arxiv.org/abs/1802.00420)
7. Xu and Singh. *Robust Universal Adversarial Perturbations*. ICML 2024. [arXiv:2206.10858](https://arxiv.org/abs/2206.10858)
8. P. Benz et al. *Adversarial Robustness Comparison of Vision Transformer and MLP-Mixer to CNNs*. 2021. [arXiv:2110.02797](https://arxiv.org/abs/2110.02797)
9. ImageNetV2 dataset: [Hugging Face](https://huggingface.co/datasets/vaishaal/ImageNetV2)