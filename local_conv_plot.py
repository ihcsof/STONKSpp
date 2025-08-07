import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# 1) Load only MAD runs, skip any convergences before iteration 5
df = pd.read_csv('local_conv_mad.csv')
df = df[df['iter'] >= 5]

# 2) Compute community size from the 'subgraph' list
df['subgraph_size'] = df['subgraph'].apply(lambda s: len(eval(s)))

# 3) Prepare unique coords and colors
t_vals = sorted(df['t'].unique(), key=lambda x: float('inf') if x == 'inf' else float(x))
xticks = sorted(df['p_attack'].unique())
unique_subs = df['subgraph'].unique()
cmap = plt.cm.get_cmap('tab20', len(unique_subs))
color_map = {sub: cmap(i) for i, sub in enumerate(unique_subs)}

offset_amount = 0.01  # horizontal shift for near‐overlaps

for tval in t_vals:
    df_t = df[df['t'] == tval]
    fig, ax = plt.subplots(figsize=(6,6))
    
    # For each x, detect clusters of y within ±3 and offset them
    for x in xticks:
        df_x = df_t[df_t['p_attack'] == x].copy()
        if df_x.empty: continue
        df_x.sort_values('iter', inplace=True)
        y = df_x['iter'].values
        offs = np.zeros(len(df_x))
        
        # Build clusters where consecutive diffs <= 3
        clusters = []
        start = 0
        for i in range(len(y)-1):
            if abs(y[i+1] - y[i]) > 3:
                if i - start + 1 > 1:
                    clusters.append(list(range(start, i+1)))
                start = i+1
        if len(y) - start > 1:
            clusters.append(list(range(start, len(y))))
        
        # Alternate offsets
        for group in clusters:
            for idx, pos in enumerate(group):
                offs[pos] = (-offset_amount if idx % 2 == 0 else offset_amount)
        
        # Plot
        for idx, row in df_x.reset_index().iterrows():
            xx = row['p_attack'] + offs[idx]
            yy = row['iter']
            size = row['subgraph_size'] * 117
            sub = row['subgraph']
            ax.scatter(xx, yy, s=size,
                       color=color_map[sub],
                       alpha=0.5, edgecolors='w', linewidth=0.7)
            ax.text(xx, yy, str(row['subgraph_size']),
                    ha='center', va='center', fontsize=12, color='black')
    
    # Axis formatting
    ax.set_xlabel('Attack probability ($p_{\\mathrm{attack}}$)')
    ax.set_ylabel('Iterations to converge')
    ax.set_title(f'Local convergence communities (t = {tval})')
    ax.set_yscale('log')
    ax.set_xticks(xticks)
    
    # Force y‐ticks at every 100 up to the max
    max_iter = int(df_t['iter'].max() + 50)
    ax.set_yticks(range(100, max_iter+1, 100))
    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    fmt.set_useOffset(False)
    ax.yaxis.set_major_formatter(fmt)
    
    ax.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.show()

