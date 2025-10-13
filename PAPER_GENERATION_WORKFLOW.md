# Paper Generation Workflow

Complete workflow for generating simulation results, tables, and figures for the paper.

## Directory Structure

```
Code/
├── Refactor2/                    # Main simulation code
│   ├── main.py                   # Run simulations
│   ├── config.yaml               # Simulation configuration
│   └── src/                      # Source code modules
├── paper_outputs/                # Simulation results (CSV files)
│   ├── results_tseries_bgn.csv
│   ├── results_fama_bgn.csv
│   ├── results_dkkm_bgn.csv
│   ├── results_ipca_bgn.csv
│   └── ... (same for kp and gs)
├── Tex/
│   ├── Tables (main)/for tex/    # Generated LaTeX tables
│   └── Images (main)/for tex/    # Generated PDF figures
├── generate_tables.py            # Generate LaTeX tables
├── generate_figures.py           # Generate PDF figures
└── generate_paper_outputs.py    # Master script (runs both)
```

## Complete Pipeline

### Step 1: Run Simulations

Generate simulation results for all three economic models (BGN, KP, GS):

```bash
# BGN model
cd Refactor2
python main.py --config config.yaml --model bgn

# KP model
python main.py --config config.yaml --model kp

# GS model
python main.py --config config.yaml --model gs
```

**Output:** CSV files in `paper_outputs/` directory

**Note:** The `config.yaml` file is already configured to output to `paper_outputs/`

### Step 2: Generate Tables and Figures

Once all simulations are complete, generate paper outputs:

```bash
cd ..  # Return to Code directory
python generate_paper_outputs.py
```

This master script runs both:
- `generate_tables.py` - Creates LaTeX table files
- `generate_figures.py` - Creates PDF figure files

**Output:**
- Tables: `Tex/Tables (main)/for tex/*.tex`
- Figures: `Tex/Images (main)/for tex/*.pdf`

### Step 3: Compile Paper

```bash
cd Tex
pdflatex main.tex
```

## Individual Scripts

### Generate Tables Only

```bash
python generate_tables.py
```

**Generates:**
- `Tex/Tables (main)/for tex/fama_comparisons.tex`
- `Tex/Tables (main)/for tex/dkkm_comparisons.tex`
- `Tex/Tables (main)/for tex/kps_comparisons.tex`
- `Tex/Tables (main)/for tex/sdf_weight_statistics_2.tex` (placeholder)

### Generate Figures Only

```bash
python generate_figures.py
```

**Generates:**
- `Tex/Images (main)/for tex/dkkm_bgn_ver3.pdf`
- `Tex/Images (main)/for tex/dkkm_kp_ver3.pdf`
- `Tex/Images (main)/for tex/dkkm_gs_ver3.pdf`
- `Tex/Images (main)/for tex/ipca_bgn_ver3.pdf`
- `Tex/Images (main)/for tex/ipca_kp_ver3.pdf`
- `Tex/Images (main)/for tex/ipca_gs_ver3.pdf`

## Simulation Configuration

Edit `Refactor2/config.yaml` to configure simulations:

```yaml
# Model selection (bgn, kp, or gs)
model: bgn

# Simulation parameters
start_iter: 0
num_iters: 100        # Number of simulation iterations

# Panel dimensions
N: 100                # Number of firms
T: 720                # Number of time periods (60 years monthly)
burnin: 200           # Burnin period

# DKKM parameters
include_mkt: true
nmat: 5               # Number of random weight matrices
nfeatures_lst: [6, 36, 360, 3600]  # Feature counts

# Regularization parameters
alpha_lst_fama: [0]
alpha_lst: [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1]

# IPCA parameters
ipca_nfactors_lst: [1, 2, 3]

# Computation parameters
n_jobs: 10            # Number of parallel CPU cores

# Output directory
output_dir: paper_outputs
```

## Data Flow

```
Simulation (main.py)
    ↓
CSV Results (paper_outputs/)
    ↓
    ├─→ generate_tables.py → LaTeX tables (Tex/Tables (main)/for tex/)
    └─→ generate_figures.py → PDF figures (Tex/Images (main)/for tex/)
        ↓
    main.tex (compiles everything)
        ↓
    main.pdf (final paper)
```

## Results Files

### Time Series Results
- `results_tseries_{model}.csv` - Monthly time series of SDF returns and max Sharpe ratios

### Fama Results
- `results_fama_{model}.csv` - Fama-French (FF) and Fama-MacBeth (FM) results

### DKKM Results
- `results_dkkm_{model}.csv` - DKKM method results with various κ and feature counts

### IPCA Results
- `results_ipca_{model}.csv` - KPS/IPCA method results with various factor counts

## Troubleshooting

### Missing CSV Files

If you get an error about missing CSV files:

```
ERROR: Missing required CSV files:
  - paper_outputs/results_tseries_bgn.csv
```

**Solution:** Run simulations for the missing model:
```bash
cd Refactor2
python main.py --config config.yaml --model bgn
```

### Empty paper_outputs/ Directory

If the `paper_outputs/` directory doesn't exist:

**Solution:** It will be created automatically when you run simulations. Make sure `config.yaml` has:
```yaml
output_dir: paper_outputs
```

### Figure Generation Errors

If figure generation fails with matplotlib/seaborn errors:

**Solution:** Install required packages:
```bash
pip install matplotlib seaborn pandas numpy scipy
```

### Table Generation Errors

If table generation fails:

**Solution:** Check that all CSV files exist and contain the expected columns:
- `iter`, `month` (identifiers)
- `mn`, `stdev` (for Sharpe ratios)
- `sdf_ret`, `xret` (for HJ distances)

## Performance Notes

### Simulation Runtime

With `n_jobs=10` on a 16-core Threadripper:
- **Single iteration:** ~30 minutes
- **100 iterations:** ~50 hours per model
- **All 3 models (300 iterations total):** ~150 hours (6.25 days)

### GPU Acceleration (Optional)

For faster simulations, see `GPU_STRATEGY_COMPLETE.pdf` for GPU migration strategy.

Expected speedup: **21.8x** (30 min → 84 sec per iteration)

## Quick Reference

```bash
# Full pipeline (all models)
cd Refactor2
for model in bgn kp gs; do
    python main.py --config config.yaml --model $model
done
cd ..
python generate_paper_outputs.py
cd Tex
pdflatex main.tex
```

## File Locations

| Item | Location |
|------|----------|
| Simulation code | [Refactor2/main.py](Refactor2/main.py) |
| Configuration | [Refactor2/config.yaml](Refactor2/config.yaml) |
| Table generation | [generate_tables.py](generate_tables.py) |
| Figure generation | [generate_figures.py](generate_figures.py) |
| Master script | [generate_paper_outputs.py](generate_paper_outputs.py) |
| Results (CSV) | [paper_outputs/](paper_outputs/) |
| Tables (LaTeX) | [Tex/Tables (main)/for tex/](Tex/Tables (main)/for tex/) |
| Figures (PDF) | [Tex/Images (main)/for tex/](Tex/Images (main)/for tex/) |
| Paper (LaTeX) | [Tex/main.tex](Tex/main.tex) |
