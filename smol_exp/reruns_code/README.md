# Re-runs: middle-layer loss and extra transfer checks

Scripts behind `smol_exp/13_middle_layer_loss_and_transfer.ipynb`. Results (JSON, captions) and the
trained perturbations (`v_*.npy`, same format as `uap_smolvlm256m_linf10.npy`) are in
`smol_exp/uap_smolvlm/reruns/`.

| File | What it does |
|---|---|
| `common.py` | data split (same as notebooks 01–07), layer-by-layer encoder forward (`encoder_states`), training (`train_uap`, `TARGETS`), real-pipeline captioning, metrics |
| `train.py` | `python train.py final h6 multi ...`: one perturbation per loss target |
| `evaluate.py` | captions changed on SmolVLM 256M / 500M / 2.2B (splitting off / on) + layer-wise damage |
| `transfer_other.py` | 256M vs 500M layer similarity; transfer to Google SigLIP (zero-shot), ResNet-50, ViT-B/16 |
| `robustness.py` | damage after preprocessing vs through the real pipeline |
| `s0_check.py`, `s1_e0.py` | sanity checks and reproduction of the group's numbers |
| `run_all.sh`, `run_extra.sh` | the full sequence that was run |
| `collect.py`, `build_report.py`, `report_template.html`, `make_notebook.py` | summary table, HTML report, notebook 13 |

Paths at the top of `common.py` (`DATA_DIR`, `GROUP_V`) point to the machine the runs were made on;
change them before re-running. Requirements: `torch`, `transformers>=4.50`, `sentence-transformers`,
`sentencepiece`, `torchvision`, `scipy`.
