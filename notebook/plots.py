# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns

# # ------------------------------------------------------------------------
# # 1) Set global styles (optional) and define a helper function
# # ------------------------------------------------------------------------

# # Set Seaborn style for better aesthetics (optional)
# sns.set_style("whitegrid")
# sns.set_context("paper", font_scale=1.2)

# def get_mean_std(df, arch, model, metric):
#     """
#     Retrieve mean and standard deviation for a given
#     (Architecture, Model, Metric) from the DataFrame.
#     """
#     row = df[(df['Architecture'] == arch) & (df['model'] == model)]
#     if row.empty:
#         return 0, 0
#     mean_val = row[f'{metric}_mean'].values[0]
#     std_val  = row[f'{metric}_std'].values[0]
#     return mean_val, std_val

# # ------------------------------------------------------------------------
# # 2) Define plot functions
# # ------------------------------------------------------------------------

# def plot_bar_subplots(combined_df, metrics, unique_archs, unique_models, pastel_colors, save_path):
#     """
#     Creates a figure with three subplots (one per metric) in bar chart style.
#     Each bar group corresponds to one architecture; each bar within the group
#     corresponds to a particular model.
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

#     for idx, metric in enumerate(metrics):
#         ax = axes[idx]
#         x = np.arange(len(unique_archs))  
#         width = 0.8 / len(unique_models)
        
#         for i, model in enumerate(unique_models):
#             bar_positions = x + (i - (len(unique_models)-1)/2)*width
            
#             means = []
#             stds = []
#             for arch in unique_archs:
#                 mean_val, std_val = get_mean_std(combined_df, arch, model, metric)
#                 means.append(mean_val)
#                 stds.append(std_val)
            
#             ax.bar(
#                 bar_positions,
#                 means,
#                 yerr=stds,
#                 width=width,
#                 color=pastel_colors[i % len(pastel_colors)],
#                 label=model,
#                 capsize=5
#             )
            
#         ax.set_title(metric, fontsize=12, fontweight='bold')
#         ax.set_xticks(x)
#         ax.set_xticklabels(unique_archs, rotation=45, ha='right')
#         ax.set_ylim(0, 1)  # Assuming metrics are in [0, 1]
#         ax.grid(axis='y', linestyle='--', alpha=0.7)
        
#         if idx == 0:
#             ax.set_ylabel("Value")

#     handles, labels = axes[-1].get_legend_handles_labels()
#     fig.legend(handles, labels, title='Model', loc='upper right', bbox_to_anchor=(1.08, 1.0))
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, 'all_metrics_plot.pdf'), format='pdf', bbox_inches='tight')
#     plt.show()


# def plot_bar_subplots_publication_style(combined_df, metrics, unique_archs, unique_models, model_colors, save_path):
#     """
#     Creates a publication-style bar chart with multiple subplots (one per metric),
#     annotations, etc.
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
#     plt.suptitle("Model Performance Across Architectures", y=1.02, fontsize=16, fontweight='bold')

#     for idx, metric in enumerate(metrics):
#         ax = axes[idx]
#         x = np.arange(len(unique_archs))
#         width = 0.8 / len(unique_models)
        
#         for i, model in enumerate(unique_models):
#             bar_positions = x + (i - (len(unique_models)-1)/2)*width
#             means = []
#             stds = []
#             for arch in unique_archs:
#                 mean_val, std_val = get_mean_std(combined_df, arch, model, metric)
#                 means.append(mean_val)
#                 stds.append(std_val)
            
#             bars = ax.bar(
#                 bar_positions,
#                 means,
#                 width=width,
#                 yerr=stds,
#                 color=model_colors[i],
#                 label=model,
#                 capsize=5,
#                 error_kw={'elinewidth': 1.5}
#             )
            
#             # Add value annotations on each bar
#             for pos, mean in zip(bar_positions, means):
#                 ax.text(
#                     pos, mean + 0.02, 
#                     f'{mean:.2f}', 
#                     ha='center', 
#                     va='bottom', 
#                     fontsize=10,
#                     rotation=90
#                 )

#         ax.set_title(metric, fontsize=14, pad=20)
#         ax.set_xticks(x)
#         ax.set_xticklabels(unique_archs, rotation=45, ha='right', fontsize=12)
#         ax.set_ylim(0, 1.1)
#         ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
#         ax.tick_params(axis='y', labelsize=12)
#         ax.grid(True, linestyle='--', alpha=0.6)
        
#         if idx == 0:
#             ax.set_ylabel("Score (%)", fontsize=14)
    
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(
#         handles, labels,
#         title='Models',
#         loc='upper right',
#         bbox_to_anchor=(0.95, 0.88),
#         frameon=True,
#         fontsize=12,
#         title_fontsize=13
#     )

#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, 'improved_metrics_plot.pdf'), format='pdf', bbox_inches='tight')
#     plt.show()


# def plot_point_plots(combined_df, metrics, unique_archs, unique_models, model_colors, save_path):
#     """
#     Creates a set of point plots with error bars for each metric,
#     across architectures and models.
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
#     plt.suptitle("Model Performance Comparison", y=1.02, fontsize=16, fontweight='bold')

#     for idx, metric in enumerate(metrics):
#         ax = axes[idx]
        
#         for i, model in enumerate(unique_models):
#             model_data = combined_df[combined_df['model'] == model]
#             means = []
#             stds = []
            
#             for arch in unique_archs:
#                 mean_val, std_val = get_mean_std(model_data, arch, model, metric)
#                 means.append(mean_val)
#                 stds.append(std_val)
            
#             x = np.arange(len(unique_archs))
#             ax.errorbar(
#                 x, means, yerr=stds,
#                 fmt='o', ms=8, capsize=5,
#                 color=model_colors[i],
#                 label=model,
#                 linestyle='-',
#                 linewidth=2,
#                 alpha=0.8
#             )
        
#         ax.set_title(metric, fontsize=14)
#         ax.set_xticks(x)
#         ax.set_xticklabels(unique_archs, rotation=45, ha='right', fontsize=12)
#         ax.set_ylim(0, 1.1)
#         ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
#         ax.grid(True, linestyle='--', alpha=0.6)
        
#         if idx == 0:
#             ax.set_ylabel("Score (%)", fontsize=14)
#             ax.legend(loc='upper left', fontsize=12)

#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, 'point_plot_metrics.pdf'), format='pdf', bbox_inches='tight')
#     plt.show()


# def plot_heatmaps(combined_df, metrics, save_path):
#     """
#     Creates heatmaps of mean metric values for each (model, architecture) pair.
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(20, 8))
#     plt.suptitle("Performance Heatmaps", y=1.05, fontsize=16, fontweight='bold')

#     for idx, metric in enumerate(metrics):
#         ax = axes[idx]
        
#         # Pivot data for heatmap
#         heatmap_data = combined_df.pivot_table(
#             index='model', 
#             columns='Architecture', 
#             values=f'{metric}_mean'
#         )
        
#         sns.heatmap(
#             heatmap_data,
#             annot=True,
#             fmt=".2f",
#             cmap="YlGnBu",
#             vmin=0,
#             vmax=1,
#             ax=ax,
#             cbar_kws={'label': 'Score', 'format': plt.FuncFormatter(lambda x, _: f"{x:.0%}")},
#             annot_kws={'size': 12}
#         )
        
#         ax.set_title(metric, fontsize=14)
#         ax.set_xlabel("Architecture", fontsize=12)
#         ax.set_ylabel("Model", fontsize=12)
#         ax.tick_params(axis='both', labelsize=10)

#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, 'heatmap_metrics.pdf'), format='pdf', bbox_inches='tight')
#     plt.show()

# # ------------------------------------------------------------------------
# # 3) Main script logic
# # ------------------------------------------------------------------------
# if __name__ == "__main__":
#     # Define the directory path where CSV files are located
#     directory = r"C:\Users\Emanuele\Desktop\Smartphone\SimpleCLF"
    
#     # List of CNN architectures
#     architectures = ["ConvNeXtSmall", "ResNet50", "EfficientNetV2S", "InceptionV3"]
    
#     # Construct filenames based on the architecture names
#     filenames = [os.path.join(directory, f"Overall_Statistics_{arch}.csv") for arch in architectures]
    
#     # Initialize an empty list to hold DataFrames from each CSV file
#     dfs = []
    
#     # Read each CSV file, extract architecture name, and add it as a column
#     for filename, arch in zip(filenames, architectures):
#         df = pd.read_csv(filename)
#         df['Architecture'] = arch
#         dfs.append(df)
    
#     # Concatenate all DataFrames into a single DataFrame
#     combined_df = pd.concat(dfs, ignore_index=True)
    
#     # Sort the DataFrame by Architecture and model for consistent plotting order
#     combined_df = combined_df.sort_values(by=['Architecture', 'model']).reset_index(drop=True)
    
#     # Capitalize model names for better readability (e.g., 'knn' to 'KNN')
#     combined_df['model'] = combined_df['model'].str.upper()
    
#     # Define the metrics to plot
#     metrics = ['Accuracy', 'Sensitivity', 'Specificity']
    
#     # Get unique architectures and models in the sorted order they appear
#     unique_archs = combined_df['Architecture'].unique()
#     unique_models = combined_df['model'].unique()
    
#     # Pastel color palette for models (extend if you have more than 4 models)
#     pastel_colors = [
#         "#AEC6CF",  # Pastel blue
#         "#FFB347",  # Pastel orange
#         "#B39EB5",  # Pastel purple
#         "#77DD77"   # Pastel green
#     ]
    
#     # Define colors using a publication-friendly palette
#     # Adjust n_colors=len(unique_models) if you have more models
#     model_colors = sns.color_palette("Set2", n_colors=len(unique_models))
    
#     # ------------------------------------------------------------------------
#     # Call each plotting function
#     # ------------------------------------------------------------------------
    
#     # 1) Basic bar subplots
#     plot_bar_subplots(
#         combined_df,
#         metrics,
#         unique_archs,
#         unique_models,
#         pastel_colors,
#         save_path=directory
#     )
    
#     # 2) Publication-style bar subplots
#     plot_bar_subplots_publication_style(
#         combined_df,
#         metrics,
#         unique_archs,
#         unique_models,
#         model_colors,
#         save_path=directory
#     )
    
#     # 3) Point plots
#     plot_point_plots(
#         combined_df,
#         metrics,
#         unique_archs,
#         unique_models,
#         model_colors,
#         save_path=directory
#     )
    
#     # 4) Heatmaps
#     plot_heatmaps(
#         combined_df,
#         metrics,
#         save_path=directory
#     )

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data from the provided CSV files
data = {
    'ConvNeXtSmall': {
        'model': ['knn', 'nn', 'rf', 'xgb'],
        'Accuracy': [0.6063, 0.6160, 0.6320, 0.6158],
        'Specificity': [0.7464, 0.6978, 0.7781, 0.7199],
        'Sensitivity': [0.3915, 0.4906, 0.4080, 0.4563]
    },
    'EfficientNetV2S': {
        'model': ['knn', 'nn', 'rf', 'xgb'],
        'Accuracy': [0.5820, 0.5781, 0.6070, 0.5750],
        'Specificity': [0.7771, 0.6555, 0.8764, 0.7227],
        'Sensitivity': [0.2827, 0.4594, 0.1938, 0.3485]
    },
    'InceptionV3': {
        'model': ['knn', 'nn', 'rf', 'xgb'],
        'Accuracy': [0.5949, 0.5977, 0.6166, 0.5851],
        'Specificity': [0.7842, 0.6624, 0.8773, 0.7351],
        'Sensitivity': [0.3045, 0.4986, 0.2169, 0.3551]
    },
    'ResNet50': {
        'model': ['knn', 'nn', 'rf', 'xgb'],
        'Accuracy': [0.5864, 0.5959, 0.6150, 0.5934],
        'Specificity': [0.7469, 0.6630, 0.8386, 0.7201],
        'Sensitivity': [0.3404, 0.4930, 0.2721, 0.3992]
    }
}

# Set up the plot
plt.figure(figsize=(16, 10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Create subplot for each architecture
for i, (arch, arch_data) in enumerate(data.items(), 1):
    plt.subplot(2, 2, i)
    
    # Prepare data for plotting
    x = range(len(arch_data['model']))
    width = 0.25
    
    # Plot bars
    plt.bar([pos - width for pos in x], arch_data['Accuracy'], width, label='Accuracy', color=colors[0], alpha=0.7)
    plt.bar([pos for pos in x], arch_data['Specificity'], width, label='Specificity', color=colors[1], alpha=0.7)
    plt.bar([pos + width for pos in x], arch_data['Sensitivity'], width, label='Sensitivity', color=colors[2], alpha=0.7)
    
    # Customize the plot
    plt.title(f'{arch} Performance', fontweight='bold')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.xticks(x, arch_data['model'])
    plt.ylim(0, 1)
    
    # Add legend
    plt.legend()

    # Add value labels on top of each bar
    for j, (acc, spec, sens) in enumerate(zip(arch_data['Accuracy'], arch_data['Specificity'], arch_data['Sensitivity'])):
        plt.text(j - width, acc, f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
        plt.text(j, spec, f'{spec:.3f}', ha='center', va='bottom', fontsize=8)
        plt.text(j + width, sens, f'{sens:.3f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Model Performance Across Different CNN Architectures', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()
