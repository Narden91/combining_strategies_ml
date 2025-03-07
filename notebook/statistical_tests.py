import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def load_and_merge_data(file1_path, file2_path):
    """Load and merge datasets on CNN architecture, Task, and model"""
    df1 = pd.read_csv(file1_path, delimiter=';')
    df2 = pd.read_csv(file2_path, delimiter=';')
    merged_df = pd.merge(df1, df2,
                         on=['cnn_architecture', 'Task', 'model'],
                         suffixes=('_MLP', '_SimpleCLF'),
                         how='inner')
    return merged_df

def perform_statistical_analysis(group_df, metrics, alpha=0.05):
    """Perform paired statistical tests for specified metrics"""
    results = []

    for metric in metrics:
        col1 = f"{metric}_MLP"
        col2 = f"{metric}_SimpleCLF"

        # Extract paired samples
        sample1 = group_df[col1].values
        sample2 = group_df[col2].values

        # Paired t-test
        t_stat, p_value_t = ttest_rel(sample1, sample2)

        # Wilcoxon signed-rank test
        w_stat, p_value_w = wilcoxon(sample1, sample2)

        # Determine significance
        sig_t = "Different" if p_value_t < alpha else "Similar"
        sig_w = "Different" if p_value_w < alpha else "Similar"

        results.append({
            'Metric': metric,
            'Paired t-test p-value': p_value_t,
            't-test Result': sig_t,
            'Wilcoxon p-value': p_value_w,
            'Wilcoxon Result': sig_w,
            'Sample Size': len(sample1)
        })

    return pd.DataFrame(results)

def main():
    # Directly specify file paths and metrics
    file1_path = r'C:\Users\Emanuele\Desktop\Smartphone Data Experiments\Experiment_3_ML_Classifiers\Task_Analysis_MLP.csv'
    file2_path = r'C:\Users\Emanuele\Desktop\Smartphone Data Experiments\Experiment_3_ML_Classifiers\Task_Analysis_SimpleCLF.csv'
    metrics = ['Accuracy_mean', 'Precision_mean', 'F1_mean']
    alpha = 0.05

    # Load and merge data
    merged_df = load_and_merge_data(file1_path, file2_path)
    
    print(merged_df)

    if merged_df.empty:
        print("No common records found between the files")
        return

    # Group by cnn_architecture and model
    grouped = merged_df.groupby(['cnn_architecture', 'model'])

    all_results = []

    # Perform analysis for each group
    for (architecture, model), group_df in grouped:
        results_df = perform_statistical_analysis(group_df, metrics, alpha)
        results_df['cnn_architecture'] = architecture
        results_df['model'] = model
        all_results.append(results_df)

    # Combine all results
    final_results_df = pd.concat(all_results, ignore_index=True)
    final_results_df.to_csv('statistical_comparison_results.csv', index=False)

    # Print results
    print("\nStatistical Comparison Results:")
    print(f"Significance level: {alpha}")
    print(f"Number of compared pairs: {len(merged_df)}")
    print(f"Compared metrics: {', '.join(metrics)}")
    print("\nDetailed results:")
    print(final_results_df.to_string(index=False))

if __name__ == "__main__":
    main()
