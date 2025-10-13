import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Assuming tbl is already created as shown in the notebook
folder = 'new_results' 
df = None
for theory in ['bgn', 'kp', 'gs']:
    sdf = pd.read_csv(f"{folder}/results_tseries_{theory}.csv")[['iter', 'month', 'sdf_ret']]
    for method in ['dkkm', 'fama', 'ipca', 'model']:
        data = pd.read_csv(f"{folder}/results_{method}_{theory}.csv")
        data['theory'] = theory
        data['factor_method'] = method 
        data = data.merge(sdf, on = ['iter', 'month'])
        data['sq_diff'] = (data.xret-data.sdf_ret)**2
        df = pd.concat((df, data))      

df2 = df[(df.factor_method=='dkkm') & (df.theory=='bgn')]
tbl = df2.groupby(['alpha', 'nfeatures', 'method', 'include_mkt']).sq_diff.mean().unstack().unstack()**(1/2)
tbl = tbl.loc[0.01:1.0]

# Reshape the data for plotting
plot_data = []
for alpha in tbl.index.get_level_values('alpha').unique():
    for nfeatures in tbl.index.get_level_values('nfeatures').unique():
        row = tbl.loc[(alpha, nfeatures)]
        plot_data.append({
            'alpha': alpha,
            'nfeatures': nfeatures,
            'nors_0': row[(0.0, 'nors')],
            'rs_0': row[(0.0, 'rs')],
            'nors_1': row[(1.0, 'nors')],
            'rs_1': row[(1.0, 'rs')]
        })

plot_df = pd.DataFrame(plot_data)

# Create the scatter plot
fig, ax = plt.subplots(figsize=(10, 8))

# Define shapes for each column combination
markers = {
    'nors_0': 'o',  # circle for include_mkt=0, method=nors
    'rs_0': 's',    # square for include_mkt=0, method=rs
    'nors_1': '^',  # triangle up for include_mkt=1, method=nors
    'rs_1': 'v'     # triangle down for include_mkt=1, method=rs
}

# Define labels for legend
labels = {
    'nors_0': 'No Market, No RS',
    'rs_0': 'No Market, RS',
    'nors_1': 'Market, No RS',
    'rs_1': 'Market, RS'
}

# Use a colormap for values
vmin = tbl.values.min()
vmax = tbl.values.max()
norm = plt.Normalize(vmin=vmin, vmax=vmax)
cmap = cm.viridis

# Plot each column with different shapes
for col in ['nors_0', 'rs_0', 'nors_1', 'rs_1']:
    scatter = ax.scatter(plot_df['nfeatures'], 
                        plot_df['alpha'], 
                        c=plot_df[col], 
                        marker=markers[col],
                        s=100,
                        cmap=cmap,
                        norm=norm,
                        label=labels[col],
                        alpha=0.7,
                        edgecolors='black',
                        linewidth=0.5)

# Set x-axis to log scale since nfeatures spans a wide range
ax.set_xscale('log')

# Labels and title
ax.set_xlabel('Number of Features', fontsize=12)
ax.set_ylabel('Alpha', fontsize=12)
ax.set_title('RMSE by Alpha and Number of Features', fontsize=14)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('RMSE Value', fontsize=11)

# Add legend for shapes
ax.legend(loc='best', fontsize=10)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('scatter_plot.pdf', dpi=300, bbox_inches='tight')
plt.savefig('scatter_plot.png', dpi=300, bbox_inches='tight')
plt.show()