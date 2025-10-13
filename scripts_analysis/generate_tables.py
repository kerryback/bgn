"""
Generate all LaTeX tables for the paper from simulation results.

This script reads CSV files from the paper_outputs folder and generates
LaTeX tables that are included in Tex/main.tex.

Usage:
    python generate_tables.py

Output:
    - Tex/Tables (main)/for tex/fama_comparisons.tex
    - Tex/Tables (main)/for tex/dkkm_comparisons.tex
    - Tex/Tables (main)/for tex/kps_comparisons.tex
    - Tex/Tables (main)/for tex/sdf_weight_statistics_2.tex
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
RESULTS_DIR = Path("../outputs")
OUTPUT_DIR = Path("../tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load and process all results data."""
    print("Loading data from paper_outputs...")

    # Load data for all three theories
    data = {}
    for theory in ['bgn', 'kp', 'gs']:
        print(f"  Loading {theory} data...")
        data[theory] = {
            'tseries': pd.read_csv(RESULTS_DIR / f"results_tseries_{theory}.csv"),
            'fama': pd.read_csv(RESULTS_DIR / f"results_fama_{theory}.csv"),
            'dkkm': pd.read_csv(RESULTS_DIR / f"results_dkkm_{theory}.csv"),
            'ipca': pd.read_csv(RESULTS_DIR / f"results_ipca_{theory}.csv"),
        }

    print("Data loaded successfully.")
    return data

def compute_metrics(data):
    """Compute Sharpe ratios and HJ distances for all methods."""
    print("Computing metrics...")

    results = {'fama': [], 'dkkm': [], 'ipca': []}

    for theory in ['bgn', 'kp', 'gs']:
        tseries = data[theory]['tseries']

        # Process Fama-French / Fama-MacBeth
        fama = data[theory]['fama'].merge(tseries, on=['iter', 'month'])
        fama['sharpe'] = fama['mn'] / fama['stdev']
        fama['hjd'] = np.sqrt((fama['sdf_ret'] - fama['xret'])**2)
        fama_grouped = fama.groupby('method')[['sharpe', 'hjd']].mean()

        for method in fama_grouped.index:
            results['fama'].append({
                'theory': theory,
                'method': method,
                'sharpe': fama_grouped.loc[method, 'sharpe'],
                'hjd': fama_grouped.loc[method, 'hjd']
            })

        # Process DKKM
        dkkm = data[theory]['dkkm'].merge(tseries, on=['iter', 'month'])
        dkkm = dkkm.rename(columns={'alpha': 'kappa', 'include_mkt': 'market'})
        dkkm['sharpe'] = dkkm['mn'] / dkkm['stdev']
        dkkm['hjd'] = np.sqrt((dkkm['sdf_ret'] - dkkm['xret'])**2)
        dkkm_grouped = dkkm.groupby(['kappa', 'nfeatures', 'method', 'market'])[['sharpe', 'hjd']].mean()

        for idx, row in dkkm_grouped.iterrows():
            results['dkkm'].append({
                'theory': theory,
                'kappa': idx[0],
                'nfeatures': idx[1],
                'method': idx[2],  # 'rs' or 'nors'
                'market': idx[3],
                'sharpe': row['sharpe'],
                'hjd': row['hjd']
            })

        # Process IPCA
        ipca = data[theory]['ipca'].merge(tseries, on=['iter', 'month'])
        ipca = ipca.rename(columns={'ipca factors': 'ipca_factors'})
        ipca['sharpe'] = ipca['mn'] / ipca['stdev']
        ipca['hjd'] = np.sqrt((ipca['sdf_ret'] - ipca['xret'])**2)
        ipca_grouped = ipca.groupby('ipca_factors')[['sharpe', 'hjd']].mean()

        for factors in ipca_grouped.index:
            results['ipca'].append({
                'theory': theory,
                'ipca_factors': factors,
                'sharpe': ipca_grouped.loc[factors, 'sharpe'],
                'hjd': ipca_grouped.loc[factors, 'hjd']
            })

    # Convert to DataFrames
    results = {k: pd.DataFrame(v) for k, v in results.items()}

    print("Metrics computed successfully.")
    return results

def generate_fama_table(results):
    """Generate Fama-French and Fama-MacBeth comparison table."""
    print("Generating fama_comparisons.tex...")

    fama_df = results['fama']

    # Create table comparing FMR (fm) and FFC (ff)
    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{Fama-French (FF) and Fama-MacBeth (FM) Comparison}\n"
    latex += "\\label{tab:fama}\n"
    latex += "\\begin{tabular}{lcccc}\n"
    latex += "\\toprule\n"
    latex += " & \\multicolumn{2}{c}{Sharpe Ratio} & \\multicolumn{2}{c}{HJ Distance} \\\\\n"
    latex += "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}\n"
    latex += "Model & FFC & FMR & FFC & FMR \\\\\n"
    latex += "\\midrule\n"

    for theory in ['bgn', 'kp', 'gs']:
        theory_data = fama_df[fama_df['theory'] == theory]
        ff_data = theory_data[theory_data['method'] == 'ff'].iloc[0]
        fm_data = theory_data[theory_data['method'] == 'fm'].iloc[0]

        latex += f"{theory.upper()} & {ff_data['sharpe']:.3f} & {fm_data['sharpe']:.3f} & "
        latex += f"{ff_data['hjd']:.3f} & {fm_data['hjd']:.3f} \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    # Write to file
    output_file = OUTPUT_DIR / "fama_comparisons.tex"
    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"  Saved to {output_file}")

def generate_dkkm_table(results):
    """Generate DKKM comparison table."""
    print("Generating dkkm_comparisons.tex...")

    dkkm_df = results['dkkm']

    # Focus on standardized (rs) with market, varying kappa and nfeatures
    dkkm_filtered = dkkm_df[(dkkm_df['method'] == 'rs') & (dkkm_df['market'] == True)]

    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{DKKM Performance: Sharpe Ratio Minus FMR}\n"
    latex += "\\label{tab:dkkm}\n"
    latex += "\\begin{tabular}{lcccc}\n"
    latex += "\\toprule\n"
    latex += "$\\kappa$ & 6 & 36 & 360 & 3600 \\\\\n"
    latex += "\\midrule\n"

    # Get FMR baseline for each theory
    fmr_baseline = {}
    for theory in ['bgn', 'kp', 'gs']:
        fmr_data = results['fama'][(results['fama']['theory'] == theory) &
                                    (results['fama']['method'] == 'fm')]
        fmr_baseline[theory] = fmr_data['sharpe'].iloc[0]

    # For each theory
    for theory in ['bgn', 'kp', 'gs']:
        latex += f"\\multicolumn{{5}}{{l}}{{\\textbf{{{theory.upper()}}}}} \\\\\n"

        theory_data = dkkm_filtered[dkkm_filtered['theory'] == theory]
        kappas = sorted(theory_data['kappa'].unique())

        for kappa in kappas:
            kappa_data = theory_data[theory_data['kappa'] == kappa]
            latex += f"{kappa:.4f} & "

            row_values = []
            for nf in [6, 36, 360, 3600]:
                nf_data = kappa_data[kappa_data['nfeatures'] == nf]
                if len(nf_data) > 0:
                    diff = nf_data['sharpe'].iloc[0] - fmr_baseline[theory]
                    row_values.append(f"{diff:.3f}")
                else:
                    row_values.append("--")

            latex += " & ".join(row_values) + " \\\\\n"

        if theory != 'gs':
            latex += "\\midrule\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    # Write to file
    output_file = OUTPUT_DIR / "dkkm_comparisons.tex"
    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"  Saved to {output_file}")

def generate_kps_table(results):
    """Generate KPS/IPCA comparison table."""
    print("Generating kps_comparisons.tex...")

    ipca_df = results['ipca']

    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{KPS Performance: Sharpe Ratio Minus FMR}\n"
    latex += "\\label{tab:kps}\n"
    latex += "\\begin{tabular}{lcccc}\n"
    latex += "\\toprule\n"
    latex += " & 1 Factor & 2 Factors & 3 Factors & Best \\\\\n"
    latex += "\\midrule\n"

    # Get FMR baseline
    fmr_baseline = {}
    for theory in ['bgn', 'kp', 'gs']:
        fmr_data = results['fama'][(results['fama']['theory'] == theory) &
                                    (results['fama']['method'] == 'fm')]
        fmr_baseline[theory] = fmr_data['sharpe'].iloc[0]

    for theory in ['bgn', 'kp', 'gs']:
        theory_data = ipca_df[ipca_df['theory'] == theory]

        row_values = []
        for nf in [1, 2, 3]:
            nf_data = theory_data[theory_data['ipca_factors'] == nf]
            if len(nf_data) > 0:
                diff = nf_data['sharpe'].iloc[0] - fmr_baseline[theory]
                row_values.append(f"{diff:.3f}")
            else:
                row_values.append("--")

        # Best performance
        best_sharpe = theory_data['sharpe'].max()
        best_diff = best_sharpe - fmr_baseline[theory]
        row_values.append(f"{best_diff:.3f}")

        latex += f"{theory.upper()} & " + " & ".join(row_values) + " \\\\\n"

    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"

    # Write to file
    output_file = OUTPUT_DIR / "kps_comparisons.tex"
    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"  Saved to {output_file}")

def generate_sdf_weight_table():
    """Generate SDF weight statistics table."""
    print("Generating sdf_weight_statistics_2.tex...")

    # This table requires SDF weight data which we need to check exists
    # For now, create a placeholder

    latex = "\\begin{table}[htbp]\n"
    latex += "\\centering\n"
    latex += "\\caption{SDF Weight Persistence and Cross-Sectional Correlations}\n"
    latex += "\\label{tab:sdf wt stats}\n"
    latex += "\\begin{tabular}{lcccc}\n"
    latex += "\\toprule\n"
    latex += " & True & FFC & FMR & KPS & DKKM \\\\\n"
    latex += "\\midrule\n"
    latex += "\\multicolumn{5}{l}{\\textbf{Panel I: AR(1) Persistence}} \\\\\n"
    latex += "BGN & -- & -- & -- & -- & -- \\\\\n"
    latex += "KP & -- & -- & -- & -- & -- \\\\\n"
    latex += "GS & -- & -- & -- & -- & -- \\\\\n"
    latex += "\\midrule\n"
    latex += "\\multicolumn{5}{l}{\\textbf{Panel II: Cross-Sectional Correlation}} \\\\\n"
    latex += "BGN & -- & -- & -- & -- & -- \\\\\n"
    latex += "KP & -- & -- & -- & -- & -- \\\\\n"
    latex += "GS & -- & -- & -- & -- & -- \\\\\n"
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    latex += "\n"
    latex += "% NOTE: This table requires SDF weight data.\n"
    latex += "% Check if results_sdf_weights_{theory}.csv files exist in paper_outputs/\n"

    # Write to file
    output_file = OUTPUT_DIR / "sdf_weight_statistics_2.tex"
    with open(output_file, 'w') as f:
        f.write(latex)

    print(f"  Saved to {output_file} (PLACEHOLDER - needs SDF weight data)")

def main():
    """Main execution function."""
    print("=" * 60)
    print("Generating LaTeX Tables for Paper")
    print("=" * 60)
    print()

    # Load data
    data = load_data()

    # Compute metrics
    results = compute_metrics(data)

    # Generate tables
    print()
    print("Generating tables...")
    generate_fama_table(results)
    generate_dkkm_table(results)
    generate_kps_table(results)
    generate_sdf_weight_table()

    print()
    print("=" * 60)
    print("All tables generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
