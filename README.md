# Geminie — Architectural Floor Plan Analyzer

Extracts wall geometries and detects rooms from architectural PDF floor plans using vector analysis and VLM-assisted labeling.

## Project Structure

```
├── data/              # Input PDF floor plans
├── output/            # Generated overlays, debug images (gitignored)
├── pipelines/         # Core analysis modules
│   ├── wall_extractor.py   # Deterministic wall polygon extraction
│   ├── wall_pipeline.py    # VLM-guided wall fingerprinting & matching
│   └── room_pipeline.py    # Room detection via ray-casting & polygon assembly
├── main.py            # Entry point for wall extraction
├── requirements.txt
└── .env               # ANTHROPIC_API_KEY
```

## Setup

```bash
pip install -r requirements.txt
echo "ANTHROPIC_API_KEY=sk-..." > .env
```

## Usage

**Wall extraction:**
```bash
python main.py data/your_plan.pdf
```

**Room detection (vector-first):**
```bash
python -m pipelines.room_pipeline --pdf data/your_plan.pdf --scale 96
```

**Room detection with precomputed wall vectors:**
```bash
python -m pipelines.room_pipeline \
  --pdf data/your_plan.pdf \
  --wall-vectors output/wall_vectors.json \
  --wall-vector-mode auto
```
