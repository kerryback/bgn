from data_processing import load_and_process_data

# Load and process data
dkkm, fama, ipca, dkkm_hjd_tables, dkkm_sharpe_tables = load_and_process_data()

# Function to create DKKM table with four subtables
def create_dkkm_table(theory_data, theory_name, metric_name, metric_type):
    if metric_type == 'hjd':
        # Convert squared differences to HJD
        data = theory_data.apply(lambda x: x**(1/2))
    else:
        data = theory_data
    
    latex = f"\\clearpage\n"
    latex += f"\\section{{DKKM {metric_name} - {theory_name.upper()}}}\n\n"
    latex += f"\\begin{{table}}[h]\n"
    latex += f"\\centering\n"
    latex += f"\\caption{{{theory_name.upper()} Model: {metric_name} by Regularization and Features}}\n"
    latex += f"\\begin{{tabular}}{{cc}}\n"
    
    # Define the four combinations
    combos = [
        (True, False, "RS, No Market"),
        (True, True, "RS, Market"),
        (False, False, "No RS, No Market"),
        (False, True, "No RS, Market")
    ]
    
    subtables = []
    for standardized, market, title in combos:
        if (standardized, market) in data.index:
            subset = data.loc[(standardized, market)]
            
            # Get unique nfeatures columns - handle both 'sq_diff' and 'sharpe' columns
            col_name = 'sq_diff' if metric_type == 'hjd' else 'sharpe'
            nfeatures_cols = sorted(subset.columns.get_level_values(1).unique())
            
            subtable = f"\\begin{{subtable}}{{0.48\\textwidth}}\n"
            subtable += f"\\centering\n"
            subtable += f"\\caption{{{title}}}\n"
            subtable += f"\\begin{{tabular}}{{{'c' * (len(nfeatures_cols) + 1)}}}\n"
            subtable += "\\toprule\n"
            subtable += "$\\kappa$ & " + " & ".join([f"{int(col)}" for col in nfeatures_cols]) + " \\\\\n"
            subtable += "\\midrule\n"
            
            # Get unique kappa values
            kappa_values = subset.index.unique()
            for kappa in sorted(kappa_values):
                row_values = []
                for nf in nfeatures_cols:
                    try:
                        val = subset.loc[kappa, (col_name, nf)]
                        row_values.append(f"{val:.3f}")
                    except:
                        row_values.append("--")
                # Format kappa based on theory
                if theory_name.lower() == 'gs':
                    kappa_str = f"{kappa:.7f}"
                else:
                    kappa_str = f"{kappa:.3f}"
                subtable += f"{kappa_str} & " + " & ".join(row_values) + " \\\\\n"
            
            subtable += "\\bottomrule\n"
            subtable += "\\end{tabular}\n"
            subtable += "\\end{subtable}\n"
            subtables.append(subtable)
    
    # Arrange subtables in 2x2 grid
    if len(subtables) == 4:
        latex += subtables[0] + " &\n" + subtables[1] + " \\\\[1em]\n"
        latex += subtables[2] + " &\n" + subtables[3] + "\n"
    
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

# Create Fama table
def create_fama_table(metric_name, metric_type):
    latex = f"\\clearpage\n"
    latex += f"\\section{{Fama-MacBeth {metric_name}}}\n\n"
    latex += f"\\begin{{table}}[h]\n"
    latex += f"\\centering\n"
    latex += f"\\caption{{Fama-MacBeth {metric_name}}}\n"
    latex += f"\\begin{{tabular}}{{lcc}}\n"
    latex += "\\toprule\n"
    latex += " & FF & FM \\\\\n"
    latex += "\\midrule\n"
    
    for theory in ['bgn', 'kp', 'gs']:
        theory_data = fama[fama.theory == theory]
        if metric_type == 'hjd':
            ff_val = theory_data[theory_data.method == 'ff']['sq_diff'].values[0]**(1/2)
            fm_val = theory_data[theory_data.method == 'fm']['sq_diff'].values[0]**(1/2)
        else:
            ff_val = theory_data[theory_data.method == 'ff']['sharpe'].values[0]
            fm_val = theory_data[theory_data.method == 'fm']['sharpe'].values[0]
        latex += f"{theory.upper()} & {ff_val:.3f} & {fm_val:.3f} \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

# Create IPCA table
def create_ipca_table(metric_name, metric_type):
    latex = f"\\section{{IPCA {metric_name}}}\n\n"
    latex += f"\\begin{{table}}[h]\n"
    latex += f"\\centering\n"
    latex += f"\\caption{{IPCA {metric_name}}}\n"
    latex += f"\\begin{{tabular}}{{lccc}}\n"
    latex += "\\toprule\n"
    latex += " & 1 Factor & 2 Factors & 3 Factors \\\\\n"
    latex += "\\midrule\n"
    
    for theory in ['bgn', 'kp', 'gs']:
        theory_data = ipca[ipca.theory == theory]
        vals = []
        for nf in [1, 2, 3]:
            if metric_type == 'hjd':
                val = theory_data[theory_data.ipca_factors == nf]['sq_diff'].values[0]**(1/2)
            else:
                val = theory_data[theory_data.ipca_factors == nf]['sharpe'].values[0]
            vals.append(f"{val:.3f}")
        latex += f"{theory.upper()} & " + " & ".join(vals) + " \\\\\n"
    
    latex += "\\bottomrule\n"
    latex += "\\end{tabular}\n"
    latex += "\\end{table}\n"
    
    return latex

# Create full LaTeX document with requested organization
latex_doc = r"""\documentclass[11pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{subcaption}
\usepackage{geometry}
\geometry{margin=1in}

\title{Hansen-Jagannathan Distance and Sharpe Ratio Analysis}
\author{}
\date{\today}

\begin{document}

\maketitle

\tableofcontents
\clearpage

"""

# Add HJD tables first
# (1) Fama HJD
latex_doc += create_fama_table("Hansen-Jagannathan Distance", "hjd")

# (2) IPCA HJD
latex_doc += create_ipca_table("Hansen-Jagannathan Distance", "hjd")

# (3) DKKM HJD BGN
latex_doc += create_dkkm_table(dkkm_hjd_tables['bgn'], 'bgn', "Hansen-Jagannathan Distance", "hjd")

# (4) DKKM HJD KP (note: typo in request said IP, assuming KP)
latex_doc += create_dkkm_table(dkkm_hjd_tables['kp'], 'kp', "Hansen-Jagannathan Distance", "hjd")

# (5) DKKM HJD GS
latex_doc += create_dkkm_table(dkkm_hjd_tables['gs'], 'gs', "Hansen-Jagannathan Distance", "hjd")

# Add Sharpe tables
# (6) Fama Sharpe
latex_doc += create_fama_table("Sharpe Ratio", "sharpe")

# (7) IPCA Sharpe
latex_doc += create_ipca_table("Sharpe Ratio", "sharpe")

# (8) DKKM Sharpe BGN
latex_doc += create_dkkm_table(dkkm_sharpe_tables['bgn'], 'bgn', "Sharpe Ratio", "sharpe")

# (9) DKKM Sharpe KP
latex_doc += create_dkkm_table(dkkm_sharpe_tables['kp'], 'kp', "Sharpe Ratio", "sharpe")

# (10) DKKM Sharpe GS
latex_doc += create_dkkm_table(dkkm_sharpe_tables['gs'], 'gs', "Sharpe Ratio", "sharpe")

# Close document
latex_doc += "\n\\end{document}"

# Save to file
with open("hjd_sharpe_analysis.tex", 'w') as f:
    f.write(latex_doc)

print("Complete LaTeX document saved to hjd_sharpe_analysis.tex")