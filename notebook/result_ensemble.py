import pandas as pd
from pathlib import Path
import numpy as np
import warnings

# Suppress pandas warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def extract_all_methods_metrics(base_path, classifiers_folder, cnn_architectures):
    """
    Extract metrics from ALL_METHODS CSV files for each CNN architecture and classifier.
    
    Args:
        base_path (Path): Base directory path
        classifiers_folder (Path): Subfolder containing classifier results
        cnn_architectures (list): List of CNN architecture names
    
    Returns:
        DataFrame: Combined metrics dataframe
    """
    print(f"Processing metrics files from {len(cnn_architectures)} CNN architectures...")
    
    # List of all method files and their corresponding model names
    method_files = {
        'Metrics_knn_ALL_METHODS.csv': 'knn',
        'Metrics_neural_network_ALL_METHODS.csv': 'nn',
        'Metrics_random_forest_ALL_METHODS.csv': 'rf',
        'Metrics_xgboost_ALL_METHODS.csv': 'xgb'
    }
    
    # Expected metrics columns
    metric_cols = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'MCC']
    
    # Initialize list to store all metrics
    all_metrics = []
    
    # Process each CNN architecture and classifier
    for cnn in cnn_architectures:
        for method_file, classifier in method_files.items():
            metrics_path = base_path / classifiers_folder / cnn / "ClassificationML" / method_file
            
            if not metrics_path.exists():
                print(f"File not found: {metrics_path}")
                continue
            
            # Read the CSV file
            try:
                df = pd.read_csv(metrics_path)
                
                # Extract only Mean and Std rows
                mean_rows = df[df['Run'] == 'Mean']
                std_rows = df[df['Run'] == 'Std']
                
                if mean_rows.empty or std_rows.empty:
                    print(f"No Mean/Std rows found in {metrics_path}")
                    continue
                
                # Process each method
                for method in mean_rows['Method'].unique():
                    method_means = mean_rows[mean_rows['Method'] == method]
                    method_stds = std_rows[std_rows['Method'] == method]
                    
                    # Extract metrics
                    metrics_dict = {
                        'CNN': cnn,
                        'Classifier': classifier,
                        'Method': method
                    }
                    
                    # Add mean values for each metric
                    for metric in metric_cols:
                        if metric in method_means.columns:
                            try:
                                mean_val = method_means[metric].values[0]
                                # Ensure it's numeric
                                mean_val = pd.to_numeric(mean_val, errors='coerce')
                                if not pd.isna(mean_val):
                                    # Scale non-MCC metrics by 100
                                    if metric != 'MCC':
                                        mean_val = round(mean_val * 100, 3)
                                    else:
                                        mean_val = round(mean_val, 3)
                                    metrics_dict[f'{metric}_Mean'] = mean_val
                            except:
                                pass
                    
                    # Add std values for each metric
                    for metric in metric_cols:
                        if metric in method_stds.columns:
                            try:
                                std_val = method_stds[metric].values[0]
                                # Ensure it's numeric
                                std_val = pd.to_numeric(std_val, errors='coerce')
                                if not pd.isna(std_val):
                                    # Scale non-MCC metrics by 100
                                    if metric != 'MCC':
                                        std_val = round(std_val * 100, 3)
                                    else:
                                        std_val = round(std_val, 3)
                                    metrics_dict[f'{metric}_Std'] = std_val
                            except:
                                pass
                    
                    all_metrics.append(metrics_dict)
                
                print(f"Processed {method_file} for {cnn}")
            except Exception as e:
                print(f"Error processing {metrics_path}: {e}")
    
    # Create a dataframe from all the collected metrics
    if not all_metrics:
        print("No metrics were collected!")
        return None
    
    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df

def generate_summaries(metrics_df, output_dir):
    """
    Generate summary files from the metrics dataframe.
    Uses explicit, error-resistant approaches.
    
    Args:
        metrics_df (DataFrame): The metrics dataframe
        output_dir (Path): Directory to save summaries
    """
    # Check if we have the accuracy column
    if 'Accuracy_Mean' not in metrics_df.columns:
        print("Cannot generate summaries: Accuracy_Mean column not found.")
        return
    
    # Ensure Accuracy_Mean is numeric
    metrics_df['Accuracy_Mean'] = pd.to_numeric(metrics_df['Accuracy_Mean'], errors='coerce')
    
    try:
        # 1. Find best method for each CNN-Classifier combination
        best_by_cnn_clf = []
        
        for cnn in metrics_df['CNN'].unique():
            for clf in metrics_df['Classifier'].unique():
                subset = metrics_df[(metrics_df['CNN'] == cnn) & (metrics_df['Classifier'] == clf)]
                
                # Skip if empty or no valid accuracy values
                subset = subset.dropna(subset=['Accuracy_Mean'])
                if subset.empty:
                    continue
                
                # Find best method
                best_idx = subset['Accuracy_Mean'].idxmax()
                best_row = subset.loc[best_idx].copy()
                best_by_cnn_clf.append(best_row)
        
        if best_by_cnn_clf:
            best_df = pd.DataFrame(best_by_cnn_clf)
            best_df.to_csv(output_dir / "Best_Method_By_CNN_Classifier.csv", index=False)
            print(f"Created best methods by CNN/Classifier summary")
    except Exception as e:
        print(f"Error creating best methods by CNN/Classifier summary: {e}")
    
    try:
        # 2. All methods overall (across all CNNs and classifiers), sorted by accuracy
        all_methods_overall = metrics_df.dropna(subset=['Accuracy_Mean']).sort_values(
            by='Accuracy_Mean', ascending=False
        )
        
        all_methods_overall.to_csv(output_dir / "All_Methods_Overall.csv", index=False)
        print(f"Created all methods overall summary")
    except Exception as e:
        print(f"Error creating all methods overall summary: {e}")
    
    try:
        # 3. Method performance across CNNs (for each classifier)
        for clf in metrics_df['Classifier'].unique():
            clf_data = metrics_df[metrics_df['Classifier'] == clf]
            
            if clf_data.empty:
                continue
            
            # Find unique methods for this classifier
            methods = clf_data['Method'].unique()
            
            # Create a summary for each method
            method_summaries = []
            
            for method in methods:
                method_data = clf_data[clf_data['Method'] == method]
                
                # Calculate average performance across CNNs
                if 'Accuracy_Mean' in method_data.columns:
                    accuracy_vals = pd.to_numeric(method_data['Accuracy_Mean'], errors='coerce')
                    valid_values = accuracy_vals.dropna()
                    
                    if not valid_values.empty:
                        avg_accuracy = valid_values.mean()
                        std_accuracy = valid_values.std() if len(valid_values) > 1 else 0
                        
                        method_summaries.append({
                            'Classifier': clf,
                            'Method': method,
                            'Avg_Accuracy': round(avg_accuracy, 3),
                            'Std_Across_CNNs': round(std_accuracy, 3),
                            'Num_CNNs': len(valid_values)
                        })
            
            if method_summaries:
                method_summary_df = pd.DataFrame(method_summaries)
                method_summary_df = method_summary_df.sort_values(by='Avg_Accuracy', ascending=False)
                method_summary_df.to_csv(output_dir / f"Method_Summary_{clf}.csv", index=False)
                print(f"Created method summary for {clf}")
    except Exception as e:
        print(f"Error creating method summaries by classifier: {e}")
    
    try:
        # 4. Best method for each CNN (across all classifiers)
        best_by_cnn = []
        
        for cnn in metrics_df['CNN'].unique():
            cnn_data = metrics_df[metrics_df['CNN'] == cnn]
            
            # Skip if empty or no valid accuracy values
            cnn_data = cnn_data.dropna(subset=['Accuracy_Mean'])
            if cnn_data.empty:
                continue
            
            # Find best method
            best_idx = cnn_data['Accuracy_Mean'].idxmax()
            best_row = cnn_data.loc[best_idx].copy()
            best_by_cnn.append(best_row)
        
        if best_by_cnn:
            cnn_best_df = pd.DataFrame(best_by_cnn)
            cnn_best_df.to_csv(output_dir / "Best_Method_By_CNN.csv", index=False)
            print(f"Created best methods by CNN summary")
            
            # Create a simplified summary table showing best method per CNN
            summary_rows = []
            for _, row in cnn_best_df.iterrows():
                summary_rows.append({
                    'CNN': row['CNN'],
                    'Best_Classifier': row['Classifier'],
                    'Best_Method': row['Method'],
                    'Accuracy': row['Accuracy_Mean'],
                    'MCC': row.get('MCC_Mean', 'N/A')
                })
            
            pd.DataFrame(summary_rows).to_csv(output_dir / "Best_Methods_Summary.csv", index=False)
    except Exception as e:
        print(f"Error creating best methods by CNN summary: {e}")

def main():
    # Define paths
    base_path = Path("C:/Users/Emanuele/Desktop/Smartphone")
    classifiers_folder = Path("MLP")
    cnn_architectures = ["EfficientNetV2S", "ConvNeXtSmall", "ResNet50", "InceptionV3"]
    
    # Extract metrics
    metrics_df = extract_all_methods_metrics(base_path, classifiers_folder, cnn_architectures)
    
    if metrics_df is None:
        print("Failed to extract metrics data.")
        return
    
    # Create output directory
    output_dir = base_path / classifiers_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save complete metrics data
    metrics_df.to_csv(output_dir / "All_Methods_Metrics_Summary.csv", index=False)
    print(f"Saved metrics summary to {output_dir / 'All_Methods_Metrics_Summary.csv'}")
    
    # Generate additional summaries
    generate_summaries(metrics_df, output_dir)
    
    print("Metrics processing complete!")

if __name__ == "__main__":
    main()