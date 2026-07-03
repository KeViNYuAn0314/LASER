<div align="center">

# LASER: A Corrective Lens for LVLMs via Visual Attention Preservation and Sink Suppression

### ECCV 2026

[![arXiv](https://img.shields.io/badge/arXiv-2607.01707-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.01707)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/KeViNYuAn0314/LASER)

</div>

LASER is a GRPO post-training framework that mitigates **visual forgetting** in vision–language models by shaping *when* and *where* the model attends to visual evidence during long-horizon reasoning.

---

## 📰 News

- **[2026.06]** 🎉 LASER is accepted to **ECCV 2026**.
- **[2026.06]** 🚀 Code and training scripts released.

---

## 🔍 Overview

LVLMs reason well but progressively drift away from the image during long chains of thought. We trace this *visual forgetting* to two overlooked causes:

- **When (temporal):** decay that begins in early decoding is the most damaging.
- **Where (distributional):** even when total visual attention is preserved, it collapses onto a few task-irrelevant "sink" tokens.

<p align="center">
  <img src="assets/teaser.png" width="80%" alt="Visual forgetting illustration"/>
</p>

**LASER** addresses both with two attention-derived rewards.

<p align="center">
  <img src="assets/framework.png" width="92%" alt="LASER framework"/>
</p>


---

## ⚙️ Getting Started

LASER is built on the [verl](https://github.com/volcengine/verl) RL framework and targets Qwen2.5-VL on CUDA 12.8 (PyTorch 2.8.0, vLLM 0.11.0).

**Container.** Use a vLLM 0.11.0 / CUDA 12.8 image (which already ships compiled `vllm`, `flashinfer`, and `flash-attn`), mount this repo, and install it:

```bash
git clone https://github.com/<user>/LASER.git
cd LASER
pip install -e .
```


> `transformers==4.57.6` is pinned on purpose — the hook-based attention capture mirrors Qwen2.5-VL's attention forward from that version. 


---

## 🚀 Training


### Stage 1 — Cold-start SFT (optional)

Following [Revisual-R1](https://arxiv.org/abs/2506.04207), we first cold-start `Qwen2.5-VL-7B-Instruct` with supervised fine-tuning (2 epochs) to strengthen reasoning ability before reinforcement learning. We adopt the Revisual-R1 cold-start pipeline — please refer to their official release to produce the SFT checkpoint.

### Stage 2 — GRPO with LASER rewards

Edit the paths at the top of [`train.sh`](train.sh) (or pass them as environment variables) and launch:

```bash
MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
TRAIN_FILES=/path/to/train.parquet \
VAL_FILES=/path/to/val.parquet \
N_GPUS=4 INFER_TP=2 \
bash train.sh
```

**Evaluation.** Merge the FSDP checkpoint to HuggingFace format, then evaluate with [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) (temperature 0.6, top-p 0.95):

```bash
python -m verl.model_merger merge --backend fsdp \
    --local_dir  <OUTPUT_DIR>/laser/<exp>/global_step_<N>/actor \
    --target_dir <OUTPUT_DIR>/laser/<exp>/merged_step_<N>
```

---

## 🙏 Acknowledgements

LASER is built on top of the [verl](https://github.com/volcengine/verl) reinforcement-learning framework. We thank the verl team and the authors of the datasets and baselines used in our experiments.

## 📖 Citation

```bibtex
@misc{yuan2026lasercorrectivelenslvlms,
      title={LASER: A Corrective Lens for LVLMs via Visual Attention Preservation and Sink Suppression}, 
      author={Bowen Yuan and Zijian Wang and Yadan Luo and Shijie Wang and Zi Huang},
      year={2026},
      eprint={2607.01707},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.01707}, 
}
```

## 📜 License

Released under the [Apache 2.0 License](LICENSE).
