import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

OUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(OUT_DIR, exist_ok=True)

# Load dataset
iris_df = sns.load_dataset('iris')

# 1) Sepal Length vs Sepal Width
plt.figure()
sns.scatterplot(data=iris_df, x='sepal_length', y='sepal_width', hue='species', s=80, palette='Set2')
plt.title('Sepal Length vs Sepal Width')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.tight_layout()
sep_sepal_path = os.path.join(OUT_DIR, 'sepal_length_vs_sepal_width.png')
plt.savefig(sep_sepal_path, dpi=150)
plt.close()

# 2) Petal Length vs Petal Width
plt.figure()
sns.scatterplot(data=iris_df, x='petal_length', y='petal_width', hue='species', s=80, palette='Set1')
plt.title('Petal Length vs Petal Width')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.tight_layout()
petal_path = os.path.join(OUT_DIR, 'petal_length_vs_petal_width.png')
plt.savefig(petal_path, dpi=150)
plt.close()

# 3) Sepal Length vs Petal Length
plt.figure()
sns.scatterplot(data=iris_df, x='sepal_length', y='petal_length', hue='species', s=80, palette='husl')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
sepal_petal_path = os.path.join(OUT_DIR, 'sepal_length_vs_petal_length.png')
plt.savefig(sepal_petal_path, dpi=150)
plt.close()

# 4) Histograms (all features)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
numerical_cols = iris_df.select_dtypes(include=['float64', 'int64']).columns
for idx, col in enumerate(numerical_cols):
    row = idx // 2
    col_idx = idx % 2
    ax = axes[row, col_idx]
    sns.histplot(data=iris_df, x=col, hue='species', kde=True, ax=ax, palette='Set2', bins=20)
    ax.set_title(f'Distribution of {col.replace("_", " ").title()}')
plt.tight_layout()
hist_path = os.path.join(OUT_DIR, 'feature_histograms.png')
fig.savefig(hist_path, dpi=150)
plt.close(fig)

# 5) Box plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for idx, col in enumerate(numerical_cols):
    row = idx // 2
    col_idx = idx % 2
    ax = axes[row, col_idx]
    sns.boxplot(data=iris_df, x='species', y=col, ax=ax, palette='Set2')
    ax.set_title(f'Box Plot of {col.replace("_", " ").title()}')
plt.tight_layout()
box_path = os.path.join(OUT_DIR, 'feature_boxplots.png')
fig.savefig(box_path, dpi=150)
plt.close(fig)

print('Saved plots:')
for p in [sep_sepal_path, petal_path, sepal_petal_path, hist_path, box_path]:
    print(' -', p)
