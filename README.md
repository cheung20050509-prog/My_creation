# InfoGate: Information Bottleneck-Guided Adaptive Cross-Attention for Robust Multimodal Fusion

Multimodal sentiment analysis (MOSI/MOSEI) combining ITHP-style residual fusion, IB-guided cross-attention, and CyIN-style cyclic translation.

## Requirements

- Python 3.8+
- PyTorch, transformers, scikit-learn, tqdm

```bash
pip install -r requirements.txt
```

## Data

Place `mosi.pkl` and `mosei.pkl` in `datasets/` (or symlink). See e.g. [BERT_multimodal_transformer](https://github.com/WasifurRahman/BERT_multimodal_transformer) for data format.

## Model

Use local DeBERTa-v3-base: put the model files (`config.json`, `pytorch_model.bin`, `spm.model`, `tokenizer_config.json`) in `My_creation/deberta-v3-base/`, or change the code to load from `microsoft/deberta-v3-base` (requires network).

## Train

```bash
./train.sh mosi    # or: python train.py --dataset mosi
```

## Test

```bash
./test.sh mosi checkpoints/infogate_mosi_best.pt
```

## License

MIT
