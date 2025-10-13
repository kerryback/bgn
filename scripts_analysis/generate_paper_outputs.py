"""
Master script to generate all paper outputs from simulation results.

This script runs the complete pipeline:
1. Generates LaTeX tables from simulation results
2. Generates PDF figures from simulation results

It assumes that simulation results already exist in the paper_outputs/ folder.
To generate simulation results, run: cd Refactor2 && python main.py --config config.yaml

Usage:
    python generate_paper_outputs.py

Output:
    Tables:
        - Tex/Tables (main)/for tex/fama_comparisons.tex
        - Tex/Tables (main)/for tex/dkkm_comparisons.tex
        - Tex/Tables (main)/for tex/kps_comparisons.tex
        - Tex/Tables (main)/for tex/sdf_weight_statistics_2.tex

    Figures:
        - Tex/Images (main)/for tex/dkkm_bgn_ver3.pdf
        - Tex/Images (main)/for tex/dkkm_kp_ver3.pdf
        - Tex/Images (main)/for tex/dkkm_gs_ver3.pdf
        - Tex/Images (main)/for tex/ipca_bgn_ver3.pdf
        - Tex/Images (main)/for tex/ipca_kp_ver3.pdf
        - Tex/Images (main)/for tex/ipca_gs_ver3.pdf
"""

import sys
import subprocess
from pathlib import Path

def check_results_exist():
    """Check if simulation results exist."""
    results_dir = Path("paper_outputs")

    if not results_dir.exists():
        print("ERROR: paper_outputs/ directory does not exist!")
        print("Please run simulations first:")
        print("  cd Refactor2 && python main.py --config config.yaml")
        return False

    # Check for required CSV files
    required_files = []
    for theory in ['bgn', 'kp', 'gs']:
        for result_type in ['tseries', 'fama', 'dkkm', 'ipca']:
            required_files.append(results_dir / f"results_{result_type}_{theory}.csv")

    missing_files = [f for f in required_files if not f.exists()]

    if missing_files:
        print("ERROR: Missing required CSV files:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease run simulations for all three models (bgn, kp, gs)")
        return False

    return True

def run_script(script_name, description):
    """Run a Python script and handle errors."""
    print()
    print("=" * 70)
    print(f"Running: {description}")
    print("=" * 70)
    print()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )
        print()
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print()
        print(f"ERROR: {description} failed with exit code {e.returncode}")
        return False

def main():
    """Main execution function."""
    print()
    print("=" * 70)
    print("PAPER OUTPUTS GENERATION PIPELINE")
    print("=" * 70)
    print()

    # Check if results exist
    if not check_results_exist():
        sys.exit(1)

    print("[OK] All required simulation results found")
    print()

    # Run table generation
    if not run_script("generate_tables.py", "Table generation"):
        print("\nERROR: Pipeline failed at table generation")
        sys.exit(1)

    # Run figure generation
    if not run_script("generate_figures.py", "Figure generation"):
        print("\nERROR: Pipeline failed at figure generation")
        sys.exit(1)

    # Success
    print()
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()
    print("Generated outputs:")
    print("  Tables: Tex/Tables (main)/for tex/")
    print("  Figures: Tex/Images (main)/for tex/")
    print()
    print("You can now compile the paper:")
    print("  cd Tex && pdflatex main.tex")
    print()

if __name__ == "__main__":
    main()
