# preprocessing for mixing score
import os

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/Data/mixing/'
plot_dir = '/Users/noahgreenwald/Documents/Grad_School/Lab/TNBC/plots/'

# load dataset
mixing_dict = {'Cancer_Immune-homogeneous_mixing_score.csv': 'cancer_immune',
               'Cancer_Stroma-homogeneous_mixing_score.csv': 'cancer_stroma',
               'Stroma_Immune-homogeneous_mixing_score.csv': 'stroma_immune'}

mixing_dfs = []
for filename, name in mixing_dict.items():
    current_df = pd.read_csv(os.path.join(data_dir, filename))
    current_df['mixing_type'] = name
    mixing_dfs.append(current_df)

mixing_df = pd.concat(mixing_dfs, axis=0)
mixing_df = mixing_df[~mixing_df['mixing_score'].isna()]

wide_df = mixing_df.pivot(index='fov', columns='mixing_type', values='mixing_score')

sns.scatterplot(data=wide_df, x='cancer_immune', y='cancer_stroma')
plt.savefig(os.path.join(plot_dir, 'cancer_stroma_vs_cancer_immune.png'))
plt.close()

sns.scatterplot(data=wide_df, x='cancer_immune', y='stroma_immune')
plt.savefig(os.path.join(plot_dir, 'stroma_immune_vs_cancer_immune.png'))
plt.close()

sns.scatterplot(data=wide_df, x='cancer_stroma', y='stroma_immune')
plt.savefig(os.path.join(plot_dir, 'stroma_immune_vs_cancer_stroma.png'))
plt.close()

mixing_df.to_csv(os.path.join(data_dir, 'mixing_df.csv'), index=False)

