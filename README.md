# MLOps Pipeline — Internship Technical Assessment

A deterministic, containerized Python pipeline that processes CSV financial data, computes a rolling-mean trading signal, and outputs structured JSON metrics.

## Quick Start (Docker)

```bash
# Build the image
docker build -t mlops-task .

# Run the pipeline
docker run --rm mlops-task
```

The container exits with code `0` on success, non-zero on failure.

## Local Development

### Prerequisites
- Python 3.9+
- pip

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

## CLI Usage

```
python run.py --input <csv> --config <yaml> --output <json> --log-file <log>
```

| Argument | Description |
|---|---|
| `--input` | Path to input CSV file (must contain a `close` column) |
| `--config` | Path to YAML configuration file |
| `--output` | Path for output metrics JSON |
| `--log-file` | Path for execution log file |

## Configuration (`config.yaml`)

```yaml
seed: 42      # Random seed for deterministic execution
window: 5     # Rolling mean window size
version: "v1" # Version tag for output metrics
```

## Output

### Success (`metrics.json`)

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4989,
  "latency_ms": 34,
  "seed": 42,
  "status": "success"
}
```

### Error

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Description of what went wrong"
}
```

## Pipeline Steps

1. **Configuration Loading** — Parse `config.yaml`, set random seed
2. **Data Ingestion** — Load CSV, validate `close` column exists
3. **Rolling Mean** — Compute rolling mean on `close` column
4. **Signal Generation** — `1` if close > rolling mean, else `0`
5. **Metrics Output** — Write signal rate and latency to JSON

## Project Structure

```
├── run.py                  # Main pipeline
├── config.yaml             # Pipeline configuration
├── data.csv                # Input dataset
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container definition
├── metrics.json            # Example output (from a successful run)
├── run.log                 # Example log (from a successful run)
└── README.md               # This file
```

## Error Handling

The pipeline handles: missing files, invalid CSVs, empty datasets, missing columns, and malformed config files. All errors produce a structured error JSON and non-zero exit code.

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | >=1.26,<2.0 | Random seed, numerical operations |
| `pandas` | >=2.1,<3.0 | CSV ingestion, rolling mean computation |
| `pyyaml` | >=6.0,<7.0 | YAML config file parsing |

Install all dependencies with:

```bash
pip install -r requirements.txt
```
