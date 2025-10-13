"""
Generate all figures for the paper from simulation results.

This script reads CSV files from the paper_outputs folder and generates
PDF figures that are included in Tex/main.tex.

Usage:
    python generate_figures.py

Output:
    - Tex/Images (main)/for tex/dkkm_bgn_ver3.pdf
    - Tex/Images (main)/for tex/dkkm_kp_ver3.pdf
    - Tex/Images (main)/for tex/dkkm_gs_ver3.pdf
    - Tex/Images (main)/for tex/ipca_bgn_ver3.pdf
    - Tex/Images (main)/for tex/ipca_kp_ver3.pdf
    - Tex/Images (main)/for tex/ipca_gs_ver3.pdf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configuration
RESULTS_DIR = Path("paper_outputs")
OUTPUT_DIR = Path("Tex/Images (main)/for tex")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set seaborn style
sns.set_style('white')
categorical = sns.color_palette()
sequential = sns.color_palette('rocket_r')

def load_data(model):
    """Load and process data for a specific model."""
    print(f"  Loading {model.upper()} data...")

    # Load CSV files
    fama = pd.read_csv(RESULTS_DIR / f"results_fama_{model}.csv")
    dkkm = pd.read_csv(RESULTS_DIR / f"results_dkkm_{model}.csv")
    ipca = pd.read_csv(RESULTS_DIR / f"results_ipca_{model}.csv")
    tseries = pd.read_csv(RESULTS_DIR / f"results_tseries_{model}.csv")

    # Merge with time series data
    fama = fama.merge(tseries, on=["iter", "month"])
    dkkm = dkkm.merge(tseries, on=["iter", "month"])
    ipca = ipca.merge(tseries, on=["iter", "month"])

    # Rename columns
    dkkm = dkkm.rename(columns={
        "iter": "panel",
        "alpha": "kappa",
        "nfeatures": "factors"
    })
    dkkm = dkkm[(dkkm.include_mkt == 1) & (dkkm.method == 'rs')]

    fama = fama.rename(columns={"method": "model", "iter": "panel"})
    ipca = ipca.rename(columns={"ipca factors": "factors", "iter": "panel"})

    # Calculate Sharpe ratios
    fama["sharpe"] = fama.mn / fama.stdev
    dkkm["sharpe"] = dkkm.mn / dkkm.stdev
    ipca["sharpe"] = ipca.mn / ipca.stdev

    # Calculate HJ distances
    fama["hjd_realized"] = (fama.sdf_ret - fama.xret)**2
    dkkm["hjd_realized"] = (dkkm.sdf_ret - dkkm.xret)**2
    ipca["hjd_realized"] = (ipca.sdf_ret - ipca.xret)**2

    # Extract FFC and FMR
    ff = fama[fama.model == 'ff'].reset_index()
    fm = fama[fama.model == 'fm'].reset_index()

    # Set indices
    dkkm = dkkm.reset_index()
    ipca = ipca.reset_index()

    return {
        'dkkm': dkkm,
        'ipca': ipca,
        'ff': ff,
        'fm': fm
    }

def generate_dkkm_figure(data, model):
    """Generate DKKM comparison figure for a specific model."""
    print(f"  Generating DKKM figure for {model.upper()}...")

    dkkm = data['dkkm']
    ff = data['ff']
    fm = data['fm']

    # Filter kappa values based on model
    if model == 'gs':
        lb = 0.00001
        ub = 0.001
    else:
        lb = 0.01
        ub = 0.1

    dkkm_filtered = dkkm[(dkkm['kappa'] >= lb) & (dkkm['kappa'] <= ub)]

    # Compute group means
    mean_sharpe = dkkm_filtered.groupby(['kappa', 'factors'])['sharpe'].mean().reset_index()
    mean_hjd = dkkm_filtered.groupby(['kappa', 'factors'])['hjd_realized'].mean().reset_index()
    mean_hjd['hjd_realized'] = np.sqrt(mean_hjd['hjd_realized'])

    # Create side-by-side panels
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=False)

    # Left panel: Sharpe ratios
    sns.lineplot(ax=axes[0], data=mean_sharpe, x='factors', y='sharpe',
                 hue='kappa', marker='o', linewidth=2, palette=sequential)
    axes[0].set_xlabel('Number of DKKM Factors', fontsize=12)
    axes[0].axhline(ff.sharpe.mean(), linewidth=2, label='FFC', color=categorical[0])
    axes[0].axhline(fm.sharpe.mean(), linewidth=2, label='FMR', color=categorical[2])
    axes[0].set_ylabel('Mean Sharpe Ratio', fontsize=12)
    axes[0].set_xscale('log')
    axes[0].set_title('(a) Sharpe Ratios')
    axes[0].legend(title='Kappa', fontsize=10)

    # Right panel: HJ distances
    sns.lineplot(ax=axes[1], data=mean_hjd, x='factors', y='hjd_realized',
                 hue='kappa', marker='o', linewidth=2, palette=sequential)
    axes[1].axhline(np.sqrt(ff.hjd_realized.mean()), linewidth=2, label='FFC', color=categorical[0])
    axes[1].axhline(np.sqrt(fm.hjd_realized.mean()), linewidth=2, label='FMR', color=categorical[2])
    axes[1].set_xlabel('Number of DKKM Factors', fontsize=12)
    axes[1].set_ylabel('Hansen-Jagannathan Distance', fontsize=12)
    axes[1].set_xscale('log')
    axes[1].set_title('(b) HJ Distances')
    axes[1].legend(title='Kappa', fontsize=10)

    plt.tight_layout()

    # Save figure
    output_file = OUTPUT_DIR / f"dkkm_{model}_ver3.pdf"
    plt.savefig(output_file)
    plt.close()

    print(f"    Saved to {output_file}")

def generate_ipca_figure(data, model):
    """Generate IPCA comparison figure for a specific model."""
    print(f"  Generating IPCA figure for {model.upper()}...")

    ipca = data['ipca']
    ff = data['ff']
    fm = data['fm']

    # Compute group means
    mean_sharpe = ipca.groupby(['factors'])['sharpe'].mean().reset_index()
    mean_hjd = ipca.groupby(['factors'])['hjd_realized'].mean().reset_index()
    mean_hjd['hjd_realized'] = np.sqrt(mean_hjd['hjd_realized'])

    # Create side-by-side panels
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=False)

    # Left panel: Sharpe Ratios
    sns.lineplot(ax=axes[0], data=mean_sharpe, x='factors', y='sharpe',
                 marker='o', linewidth=2, label='KPS', color=sequential[2])
    axes[0].axhline(ff.sharpe.mean(), linewidth=2, label='FFC', color=categorical[0])
    axes[0].axhline(fm.sharpe.mean(), linewidth=2, label='FMR', color=categorical[2])
    axes[0].set_xlabel('Number of KPS Factors', fontsize=12)
    axes[0].set_ylabel('Mean Sharpe Ratio', fontsize=12)
    axes[0].set_xticks(np.array([1, 2, 3]))
    axes[0].set_title('(a) Sharpe Ratios')
    axes[0].legend()

    # Right panel: HJ Distances
    sns.lineplot(ax=axes[1], data=mean_hjd, x='factors', y='hjd_realized',
                 marker='o', linewidth=2, label='KPS', color=sequential[2])
    axes[1].axhline(np.sqrt(ff.hjd_realized.mean()), linewidth=2, label='FFC', color=categorical[0])
    axes[1].axhline(np.sqrt(fm.hjd_realized.mean()), linewidth=2, label='FMR', color=categorical[2])
    axes[1].set_xlabel('Number of KPS Factors', fontsize=12)
    axes[1].set_ylabel('Hansen-Jagannathan Distance', fontsize=12)
    axes[1].set_xticks(np.array([1, 2, 3]))
    axes[1].set_title('(b) HJ Distances')
    axes[1].legend()

    plt.tight_layout()

    # Save figure
    output_file = OUTPUT_DIR / f"ipca_{model}_ver3.pdf"
    plt.savefig(output_file)
    plt.close()

    print(f"    Saved to {output_file}")

def main():
    """Main execution function."""
    print("=" * 60)
    print("Generating Figures for Paper")
    print("=" * 60)
    print()

    # Generate figures for all three models
    for model in ['bgn', 'kp', 'gs']:
        print(f"Processing {model.upper()} model...")

        # Load data
        data = load_data(model)

        # Generate figures
        generate_dkkm_figure(data, model)
        generate_ipca_figure(data, model)

        print()

    print("=" * 60)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
