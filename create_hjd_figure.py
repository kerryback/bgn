import matplotlib.pyplot as plt
import numpy as np
from data_processing import load_and_process_data as lpd
import seaborn as sns
sns.set_style("white")


# Load data
dkkm, fama, ipca, dkkm_hjd_tables, dkkm_sharpe_tables = lpd()

# Compute HJD
for d in [dkkm, fama, ipca]:
    d['hjd'] = np.sqrt(d['sq_diff'])

# Create figure with 3 rows and 2 columns with shared y-axis
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle('Hansen-Jagannathan Distances', fontsize=16)

# Define theories and colors
theories = ['bgn', 'kp', 'gs']

# Use a sequential colormap for nfeatures lines
colors = plt.cm.viridis(np.linspace(0.2, 0.8, 4))

# Left column: DKKM plots
for i, theory in enumerate(theories):
    ax = axes[i, 0]
    
    # Filter DKKM data for market=True, standardized=True
    theory_data = dkkm[(dkkm.theory == theory) & (dkkm.market == True) & (dkkm.standardized == True)]
    
    # Plot lines for each nfeatures value
    nfeatures_values = sorted(theory_data.nfeatures.unique())
        
    for j, nf in enumerate(nfeatures_values):
        nf_data = theory_data[theory_data.nfeatures == nf].sort_values('kappa')
        ax.plot(nf_data['kappa'], nf_data['hjd'], 
                    color=colors[j], marker='o', markersize=4, 
                    linewidth=2, label=f'{int(nf)} features')
    

    # Formatting
    ax.set_title(f'DKKM - {theory.upper()}')
    ax.set_xlabel('κ (log scale)')
    ax.set_xscale('log')
    ax.set_xlim(6, 3600)
    ax.set_ylabel('Hansen-Jagannathan Distance')
    ax.legend()
    
# Right column: IPCA plots
for i, theory in enumerate(theories):
    ax = axes[i, 1]
    
    # Plot IPCA data
    theory_data = ipca[ipca.theory == theory]
    if not theory_data.empty:
        ax.plot(ipca.ipca_factors, ipca.hjd, 'b-o', linewidth=2, markersize=6, label='IPCA')
        
    # Add Fama-MacBeth horizontal lines
    ff = fama[(fama.theory==theory) & (fama.method=='ff')].hjd.iloc[0]
    fm = fama[(fama.theory==theory) & (fama.method=='fm')].hjd.iloc[0]
    ax.axhline(y=ff, label='FF')
    ax.axhline(y=fm, label='FM')
    
    # Formatting
    ax.set_title(f'IPCA, FF, & FM - {theory.upper()}')
    ax.set_xlabel('Number of IPCA Factors}')
    ax.set_xlabel('Number of IPCA Factors')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xticks([1, 2, 3])

axes[0, 0].set_ylim(0.2, 0.3)
axes[1, 0].set_ylim(0.2, 0.3)
# axes[2, 0].set_ylim(0, 1)
axes[0, 1].set_ylim(0.2, 0.3)
axes[1, 1].set_ylim(0.2, 0.3)
# axes[2, 1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('hjd_comparison_figure.png', dpi=300, bbox_inches='tight')
# plt.show()  # Commented to avoid timeout

print("Figure saved as hjd_comparison_figure.png")