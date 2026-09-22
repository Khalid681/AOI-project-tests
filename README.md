<<<<<<< HEAD
# Threshold-Triggered Remote Estimation Under Packet Loss

This project reproduces the loss-free random-walk baseline from *Update Rate, Accuracy, and Age of Information in a Wireless Sensor Network* and extends it with packet erasures, ACK and no-ACK feedback, receiver prediction, data-freshness metrics, and real agricultural sensor traces.

The real-data study is a trace-driven communication simulation. The agricultural CSV provides ground-truth temperature and soil-moisture measurements. Packet losses, acknowledgements, reception events and transmission delay are simulated.

## Research questions

1. How do packet erasures change update rate, estimation error and AoI under threshold-triggered reporting?
2. When does drift-based prediction reduce estimation error at the same communication rate and AoI?
3. Can a validation-selected weight reduce the failures of full-drift prediction?
4. Do the findings hold for real temperature and soil-moisture traces from multiple agricultural sensor nodes?

## Included data

- `data/Manzano2022COMPAG_data_cleaned.csv`: 15,771 timestamped records from 12 nodes.
- `data/Manzano2022COMPAG_recording_gaps.csv`: detected recording gaps. These gaps are missing observations and must not be reported as wireless packet losses.

The code infers each node's sampling interval. Node 2 normally samples every 5 minutes, while most other nodes normally sample every 20 minutes.

## Project structure

```text
src/aoi_project/data.py          Data checks, segmentation and time splits
src/aoi_project/baseline.py      Base-paper random-walk reproduction
src/aoi_project/channel.py       Threshold trigger, erasures and ACK behavior
src/aoi_project/estimators.py    Hold, full-drift and weighted-drift estimators
src/aoi_project/metrics.py       Error, communication and freshness metrics
src/aoi_project/experiments.py   Calibration and held-out experiment sweep
configs/experiment.json          Main editable experiment configuration
run_baseline.py                  Synthetic validation entry point
run_agriculture.py               Agricultural experiment entry point
paired_analysis.py               Paired weighted-drift versus hold analysis
plot_results.py                  Publication-style comparison plots
tests/                           Unit tests for core logic
```

## Installation

Open a terminal in the project directory.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 1. Run the tests

```bash
pytest -q
```

All tests should pass before running the experiment sweep.

## 2. Reproduce the base-paper benchmark

Start with a shorter validation run:

```bash
python run_baseline.py --slots 200000
```

For the final baseline, use:

```bash
python run_baseline.py --slots 1000000
```

Outputs:

- `results/raw/baseline_validation.csv`
- `results/figures/baseline_validation.png`

The analytical formulas included for the symmetric simple random walk, where `p=q=0.5`, are:

```text
Update rate = 1 / T²
Mean AoI = (5T² + 1) / 6
MSE = (T² - 1) / 6
```

The simulation should approach these values as the number of slots increases.

## 3. Run a small agricultural experiment

```bash
python run_agriculture.py --quick
```

The quick mode uses two node-signal segments, two threshold multipliers, two loss levels and three random seeds. It confirms that the complete pipeline works.

## 4. Run the full agricultural experiment

```bash
python run_agriculture.py
```

The full configuration can require several minutes, depending on the machine. Change the settings in `configs/experiment.json` before running it.

Main configuration fields:

- `split_gap_multiplier`: a new segment starts when the interval exceeds this multiple of the nominal interval.
- `minimum_segment_records`: removes very short segments. The supplied configuration uses 100 records, leaving 80 eligible node-signal segments.
- `threshold_multipliers`: multiples of the training increment scale.
- `loss_probabilities`: simulated independent packet-erasure probabilities.
- `beta_grid`: candidate prediction weights selected on validation data.
- `seeds`: repeated paired packet-loss realizations.
- `feedback_model`: `ack` or `no_ack`.
- `delivery_delay_minutes`: fixed simulated delivery delay. The default real-trace experiment uses zero delay because the dataset contains no network-delay measurements.
- `freshness_deadline_multipliers`: deadlines relative to each node's nominal interval.

## 5. Analyse and plot results

```bash
python paired_analysis.py
python plot_results.py
```

Outputs:

- `results/raw/agriculture_experiments.csv`
- `results/raw/calibration_results.csv`
- `results/summaries/agriculture_summary.csv`
- `results/summaries/paired_comparison.csv`
- `results/figures/temp_loss_comparison.png`
- `results/figures/moist_loss_comparison.png`

## Experimental logic

### Segmentation and data leakage

The program handles each node separately. It starts a new segment after a long recording gap and splits each accepted segment chronologically:

- First 60%: training
- Next 20%: validation
- Final 20%: held-out test

Training estimates a centered linear drift and the threshold scale. Validation selects beta. The test split is used once for final evaluation.

### Trigger and ACK behavior

The sensor attempts transmission when:

```text
abs(current value - sender reference) >= threshold
```

Under `ack`, the sender reference changes only after successful delivery. Under `no_ack`, it changes after every attempt, including failed attempts.

### Estimators

```text
Last-value hold:       estimate = last received value
Full drift:            estimate = last value + v * elapsed time
Weighted drift:        estimate = last value + beta * v * elapsed time
```

Every estimator receives the same packet-delivery trace. Consequently, estimators have identical AoI and update rates within a paired run. Their reconstruction errors can differ.

### Metrics

Accuracy:

- MAE
- MSE
- RMSE
- Bias

Communication:

- Attempted update rate
- Delivered update rate
- Packet-delivery ratio

Freshness:

- Time-weighted mean AoI
- Peak AoI
- 95th-percentile AoI
- Fraction of time above the freshness deadline

## Recommended publication workflow

1. Confirm the synthetic baseline against the analytical formulas.
2. Run the quick agricultural experiment and inspect raw rows.
3. Run the full experiment with at least 30 seeds.
4. Use paired comparisons because estimators share the same packet-loss realization.
5. Report results separately for temperature and soil moisture.
6. Report conditions where weighted prediction fails as well as where it improves MSE.
7. Describe the real-data part as trace-driven simulation, not a deployed-network measurement study.

## Important limitations

- The dataset covers approximately 21.5 days and 12 nodes.
- It contains sensor values but no real packet-loss, ACK or reception logs.
- A global drift estimate may not represent daily temperature cycles.
- The current erasure model is independent Bernoulli loss. A burst-loss model is a useful later extension.
- The 95% intervals in the summary use repeated-run variability. For a final paper, also consider confidence intervals clustered by sensor or continuous segment.
=======
# AOI-project-tests
Python framework for threshold-triggered remote estimation under packet loss, comparing hold and drift-based estimators using AoI, freshness, communication cost, synthetic random walks, and real agricultural sensor data.
>>>>>>> 1a52315e84fe17a8ee75159ed6e9baa0c92982ae
