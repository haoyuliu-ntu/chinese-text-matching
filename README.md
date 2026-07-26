# Chinese Semantic Text Matching

A comparison of four text-matching approaches — Bi-Encoder (Sentence-BERT), Cross-Encoder, LLM zero-shot, and LLM LoRA SFT — on Chinese question-pair datasets, with unified evaluation (Accuracy + F1) across all methods.

## Key Results

| Method | Approach | Key Metric |
|--------|----------|------------|
| Bi-Encoder | Sentence-BERT + CosineEmbeddingLoss | val Accuracy ~75–83% |
| Bi-Encoder | Sentence-BERT + TripletLoss | val Accuracy ~75–83% |
| Cross-Encoder | Interactive BERT | Higher precision, slower |
| LLM zero-shot | DashScope (Qwen) | Baseline |
| LLM SFT | Qwen2-0.5B + LoRA (r=8) | F1 = 0.5556, parse\_fail = 0% |

### LoRA SFT Details

| Metric | Value |
|--------|-------|
| Base model | Qwen2-0.5B-Instruct (500M params) |
| LoRA rank | r = 8 |
| Trainable parameters | ~1.08M (**0.22%** of total) |
| Training time (1 epoch) | ~3.5 min (RTX 4060, 5K balanced samples) |
| Parse failure rate | 0% (output format stable) |

## Datasets

| Dataset | Size | Domain |
|----------|------|--------|
| AFQMC | 34,334 train / 4,316 val | E-commerce Q&A |
| LCQMC | 238K | Search queries |
| BQ Corpus | 86K | Community Q&A |

> Datasets are **excluded** from this repo. Run `python src/download_data.py` to fetch them automatically.

## Tech Stack

- **Models**: BERT (4-layer fast / 12-layer full), Qwen2-0.5B-Instruct
- **Training**: PyTorch, HuggingFace Transformers, LoRA (peft)
- **Loss functions**: CosineEmbeddingLoss, TripletLoss
- **Pooling**: mean pooling (Sentence-BERT convention)
- **Embeddings**: DashScope text-embedding-v3 (for LLM comparison)

## Quick Start

```bash
pip install -r requirements.txt

export DASHSCOPE_API_KEY="sk-xxx" # for LLM zero-shot comparison

# 1. Download datasets
python src/download_data.py

# 2. Explore data
python src/explore_data.py

# 3. Train Bi-Encoder
python src/train_biencoder.py # CosineEmbeddingLoss
python src/train_biencoder.py --loss triplet # TripletLoss

# 4. Train Cross-Encoder
python src/train_crossencoder.py

# 5. Evaluate
python src/evaluate.py

# 6. LLM SFT (optional)
python src_llm/train_sft.py
python src_llm/evaluate_sft.py

# 7. Compare all methods
python src/compare_methods.py
python src_llm/llm_compare.py
```

## Key Findings

1. **Bi-Encoder vs Cross-Encoder**: Bi-Encoder is faster (pre-computable vectors, suitable for recall); Cross-Encoder is more precise (suitable for reranking)
2. **Pooling**: mean pooling outperforms CLS for semantic similarity (Sentence-BERT paper finding)
3. **Layer depth tradeoff**: 4-layer BERT (~45.6M params) trains in ~2-5 min; 12-layer (~110M) is more accurate but slower
4. **LoRA SFT**: Only 0.22% trainable parameters, stable output format (0% parse failure)
5. **All methods share unified evaluation** (Accuracy + F1), enabling fair comparison without evaluation methodology bias

## Visualisations

7 charts in [`outputs/figures/`](outputs/figures/):
- Label distribution
- Character/token length distribution
- Bi-encoder similarity distributions
- Bi-encoder bad-case distribution
- Method comparison bar chart

## Project Structure

```
├── src/ # BERT implementations
│ ├── dataset.py # AFQMC loading
│ ├── model.py # Bi/Cross-Encoder definitions
│ ├── train_biencoder.py
│ ├── train_crossencoder.py
│ ├── evaluate.py
│ ├── compare_methods.py
│ ├── explore_data.py
│ ├── analyze_badcases.py
│ └── download_data.py
├── src_llm/ # LLM SFT + comparison
│ ├── train_sft.py # LoRA SFT (Qwen2-0.5B)
│ ├── evaluate_sft.py
│ └── llm_compare.py # LLM zero-shot vs SFT
├── outputs/
│ ├── figures/ # 7 PNG charts
│ └── logs/ # Training/eval JSON logs
├── ARCHITECTURE.md
├── USAGE_GUIDE.md
└── requirements.txt
```

## License

MIT
