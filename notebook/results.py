import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Initialize rich console
console = Console()


def main():
    console.print(Panel(Text("Classification Metrics Processing", style="bold cyan")))
    
    # 1) Define paths and configurations
    verbose = False
    base_path = Path("C:/Users/Emanuele/Desktop/Smartphone")
    classifiers_folder = Path("MLP")  # or "SimpleCLF", etc.
    cnn_architectures = ["EfficientNetV2S", "ConvNeXtSmall", "ResNet50", "InceptionV3"]  # List of CNN architectures
    models = ['knn', 'nn', 'rf', 'xgb']  # Machine learning models
    total_runs = 20  # Number of runs per model
    
    all_data = []  # To store all dataframes
    
    # Calculate total number of files for progress bar
    total_files = len(cnn_architectures) * len(models) * total_runs
    
    # 2) Read CSV files with progress tracking
    with Progress() as progress:
        task = progress.add_task("[cyan]Reading ClassificationML CSV Files...", total=total_files)
        
        for cnn_architecture in cnn_architectures:
            cnn_folder = Path(cnn_architecture)
            path_to_classification_ml = base_path / classifiers_folder / cnn_folder / "ClassificationML"
            
            for model in models:
                for run in range(1, total_runs + 1):
                    run_folder = f'run_{run}'
                    csv_path = path_to_classification_ml / model / run_folder / 'Metrics.csv'
                    
                    if csv_path.exists():
                        df = pd.read_csv(csv_path, delimiter=',')
                        df['cnn_architecture'] = cnn_architecture
                        df['model'] = model
                        df['run'] = run
                        all_data.append(df)
                    else:
                        console.print(f"[bold red]File not found:[/] {csv_path}")
                    
                    progress.update(task, advance=1)
    
    # 3) Concatenate all data
    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        
        # 4) Remove duplicates
        duplicates = master_df.duplicated(subset=['cnn_architecture', 'model', 'run', 'Task'], keep=False)
        if duplicates.any():
            console.print("[bold yellow]Warning: Duplicate rows found![/]")
            master_df = master_df.drop_duplicates(subset=['cnn_architecture', 'model', 'run', 'Task'], keep='first')
            console.print("[green]Duplicates removed.[/]")
        
        console.print("[bold green]ClassificationML data loaded successfully![/]")
    else:
        console.print("[bold red]No ClassificationML data was loaded.[/]")
        return
    
    # 5) Summary statistics per CNN architecture, model, and run
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'MCC']
    summary_df = master_df.groupby(['cnn_architecture', 'model', 'run'])[metrics].agg(['mean', 'std'])
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    summary_df = summary_df.reset_index()
    # Scale by 100 and round to 3 decimal places, except for MCC
    for col in summary_df.columns:
        if 'mean' in col or 'std' in col:
            if 'MCC' not in col:
                summary_df[col] = (summary_df[col] * 100).round(3)
            else:
                summary_df[col] = summary_df[col].round(3)
    
    # 6) Analysis by task, CNN architecture, and model
    task_analysis = master_df.groupby(['cnn_architecture', 'Task', 'model'])[metrics].agg(['mean', 'std'])
    task_analysis.columns = ['_'.join(col) for col in task_analysis.columns]
    task_analysis = task_analysis.sort_values(by=['cnn_architecture', 'model', 'Task']).reset_index()
    # Scale by 100 and round to 3 decimal places, except for MCC
    for col in task_analysis.columns:
        if 'mean' in col or 'std' in col:
            if 'MCC' not in col:
                task_analysis[col] = (task_analysis[col] * 100).round(3)
            else:
                task_analysis[col] = task_analysis[col].round(3)
    
    # 7) Overall statistics by CNN architecture and model (all runs combined)
    overall_stats = master_df.groupby(['cnn_architecture', 'model'])[metrics].agg(['mean', 'std'])
    overall_stats.columns = ['_'.join(col) for col in overall_stats.columns]
    overall_stats = overall_stats.reset_index()
    # Scale by 100 and round to 3 decimal places, except for MCC
    for col in overall_stats.columns:
        if 'mean' in col or 'std' in col:
            if 'MCC' not in col:
                overall_stats[col] = (overall_stats[col] * 100).round(3)
            else:
                overall_stats[col] = overall_stats[col].round(3)
    
    # 8) Best model for each task (considering both Accuracy and MCC mean)
    task_analysis['score'] = task_analysis['Accuracy_mean'] + task_analysis['MCC_mean']  # Already scaled
    best_model_per_task = task_analysis.loc[task_analysis.groupby('Task')['score'].idxmax()].drop(columns=['score'])
    # Scale score for display consistency (though dropped in final table)
    task_analysis['score'] = task_analysis['score'].round(3)
    
    # =================== NEW PART: Process All Method CSV Files ===================
    console.print(Panel(Text("All Methods Processing", style="bold blue")))
    
    # List of all method files
    method_files = ['Metrics_knn_ALL_METHODS.csv', 'Metrics_neural_network_ALL_METHODS.csv', 
                   'Metrics_random_forest_ALL_METHODS.csv', 'Metrics_xgboost_ALL_METHODS.csv']
    
    # Dictionary to map file names to model names
    file_to_model_map = {
        'Metrics_knn_ALL_METHODS.csv': 'knn',
        'Metrics_neural_network_ALL_METHODS.csv': 'nn',
        'Metrics_random_forest_ALL_METHODS.csv': 'rf',
        'Metrics_xgboost_ALL_METHODS.csv': 'xgb'
    }
    
    # Initialize dataframe for storing all methods summary
    all_methods_summary = []
    
    # Process each CNN architecture's all methods files
    with Progress() as progress:
        total_method_files = len(cnn_architectures) * len(method_files)
        method_task = progress.add_task("[blue]Processing All Methods CSV Files...", total=total_method_files)
        
        for cnn_architecture in cnn_architectures:
            cnn_folder = Path(cnn_architecture)
            path_to_classification_ml = base_path / classifiers_folder / cnn_folder / "ClassificationML"
            
            for method_file in method_files:
                model_name = file_to_model_map.get(method_file, 'unknown')
                metrics_file = path_to_classification_ml / method_file
                
                if metrics_file.exists():
                    try:
                        # Read method metrics
                        df = pd.read_csv(metrics_file)
                        
                        # Extract only the Mean and Std rows
                        mean_rows = df[df['Run'] == 'Mean']
                        std_rows = df[df['Run'] == 'Std']
                        
                        # Skip if no mean or std rows
                        if mean_rows.empty or std_rows.empty:
                            console.print(f"[yellow]No Mean/Std rows found in {metrics_file}[/]")
                            progress.update(method_task, advance=1)
                            continue
                        
                        # Expected metrics columns
                        metric_cols = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'MCC']
                        
                        # Process each method in the file
                        methods = mean_rows['Method'].unique()
                        
                        for method in methods:
                            method_mean = mean_rows[mean_rows['Method'] == method]
                            method_std = std_rows[std_rows['Method'] == method]
                            
                            # Skip if method not found in both mean and std
                            if method_mean.empty or method_std.empty:
                                continue
                            
                            # Create a row for this method
                            summary_row = {
                                'CNN': cnn_architecture,
                                'Classifier': model_name,
                                'Method': method
                            }
                            
                            # Add mean and std for each metric
                            for metric in metric_cols:
                                if metric in method_mean.columns:
                                    # Get mean value and handle potential missing values
                                    mean_val = method_mean[metric].values[0] if not method_mean[metric].isna().all() else None
                                    
                                    # Scale values (except MCC) by 100 and round
                                    if mean_val is not None and metric != 'MCC':
                                        mean_val = round(mean_val * 100, 3)
                                    elif mean_val is not None:
                                        mean_val = round(mean_val, 3)
                                    
                                    summary_row[f'{metric}_Mean'] = mean_val
                                
                                if metric in method_std.columns:
                                    # Get std value and handle potential missing values
                                    std_val = method_std[metric].values[0] if not method_std[metric].isna().all() else None
                                    
                                    # Scale values (except MCC) by 100 and round
                                    if std_val is not None and metric != 'MCC':
                                        std_val = round(std_val * 100, 3)
                                    elif std_val is not None:
                                        std_val = round(std_val, 3)
                                    
                                    summary_row[f'{metric}_Std'] = std_val
                            
                            # Add to summary
                            all_methods_summary.append(summary_row)
                        
                        console.print(f"[green]Processed {method_file} for {cnn_architecture}[/]")
                    except Exception as e:
                        console.print(f"[bold red]Error processing {metrics_file}: {str(e)}[/]")
                else:
                    console.print(f"[yellow]All methods file not found: {metrics_file}[/]")
                
                progress.update(method_task, advance=1)
    
    # Convert summary to DataFrame
    if all_methods_summary:
        methods_summary = pd.DataFrame(all_methods_summary)
        
        # Save the methods summary
        output_dir = base_path / classifiers_folder
        methods_summary.to_csv(output_dir / "All_Methods_Summary.csv", index=False)
        
        # Display summary table if verbose
        if verbose:
            print_summary_table(methods_summary, "All Methods Summary")
        
        console.print("[bold green]All Methods processing complete![/]")
    else:
        console.print("[bold yellow]No All Methods data was processed.[/]")
        methods_summary = None
    
    # Process all methods data if any was loaded
    if 'methods_summary' in locals() and methods_summary is not None and not methods_summary.empty:
        # Create a visualization of the methods summary if verbose mode is enabled
        if verbose:
            try:
                create_simple_methods_chart(methods_summary, output_dir)
            except Exception as e:
                console.print(f"[bold red]Error creating methods chart: {str(e)}[/]")
        
        # Find best method for each classifier and CNN architecture
        try:
            # Group by CNN and Classifier, find row with max Accuracy_Mean
            if 'Accuracy_Mean' in methods_summary.columns:
                best_methods = methods_summary.loc[
                    methods_summary.groupby(['CNN', 'Classifier'])['Accuracy_Mean'].idxmax()
                ]
                
                # Save best methods
                best_methods.to_csv(output_dir / "Best_Method_per_Classifier.csv", index=False)
                
                # Find overall best method per CNN architecture
                best_per_cnn = methods_summary.loc[
                    methods_summary.groupby('CNN')['Accuracy_Mean'].idxmax()
                ]
                
                best_per_cnn.to_csv(output_dir / "Best_Overall_Method_per_CNN.csv", index=False)
                
                if verbose:
                    print_summary_table(best_methods, "Best Method per Classifier and CNN")
                    print_summary_table(best_per_cnn, "Best Overall Method per CNN")
        except Exception as e:
            console.print(f"[bold red]Error finding best methods: {str(e)}[/]")
        
        console.print("[bold green]All Methods processing complete![/]")
    else:
        console.print("[bold yellow]No All Methods data was processed.[/]")
    
    # =================== CONTINUE WITH ENSEMBLE PROCESSING ===================
    # NEW PART: Process Ensemble Results
    ensemble_data = []
    
    console.print(Panel(Text("Ensemble Methods Processing", style="bold magenta")))
    
    # Process each CNN architecture's ensemble results
    for cnn_architecture in cnn_architectures:
        cnn_folder = Path(cnn_architecture)
        path_to_ensemble = base_path / classifiers_folder / cnn_folder / "Ensemble"
        metrics_file = path_to_ensemble / "Metrics_ALL_METHODS.csv"
        
        if metrics_file.exists():
            console.print(f"[green]Processing ensemble metrics for {cnn_architecture}...[/]")
            try:
                # Read ensemble metrics
                ensemble_df = pd.read_csv(metrics_file)
                
                # Clean up the data - handle empty Method entries and combine with the next row
                # Use ffill instead of fillna(method='ffill') to avoid deprecation warning
                ensemble_df['Method'] = ensemble_df['Method'].ffill()
                
                # Filter out rows that are just "Mean" or "Std" labels without actual data
                ensemble_df = ensemble_df[~ensemble_df['Run'].isin(['Std'])]
                
                # Add CNN architecture info
                ensemble_df['cnn_architecture'] = cnn_architecture
                
                # Append to ensemble data list
                ensemble_data.append(ensemble_df)
            except Exception as e:
                console.print(f"[bold red]Error processing {metrics_file}: {str(e)}[/]")
        else:
            console.print(f"[yellow]Ensemble metrics file not found for {cnn_architecture}[/]")
    
    # Process ensemble data if any was loaded
    if ensemble_data:
        # Combine all ensemble data
        ensemble_master_df = pd.concat(ensemble_data, ignore_index=True)
        
        # Convert numeric Run column entries to integers - avoiding deprecation warning
        def convert_to_numeric(x):
            try:
                return pd.to_numeric(x)
            except ValueError:
                return x
                
        ensemble_master_df['Run'] = ensemble_master_df['Run'].apply(convert_to_numeric)
        
        # Create separate dataframes for numeric runs and summary rows
        numeric_runs = ensemble_master_df[ensemble_master_df['Run'].apply(lambda x: isinstance(x, (int, float)))]
        summary_rows = ensemble_master_df[ensemble_master_df['Run'] == 'Mean']
        
        # Analyze ensemble methods performance across CNN architectures
        ensemble_metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'MCC']
        
        # Create summary statistics for ensemble methods including standard deviations
        # First, extract runs for calculating standard deviations
        ensemble_runs = ensemble_master_df[ensemble_master_df['Run'] != 'Mean']
        
        # Calculate means from the Mean rows
        ensemble_summary = summary_rows.groupby(['cnn_architecture', 'Method'])[ensemble_metrics].agg('mean')
        ensemble_summary = ensemble_summary.reset_index()
        
        # Calculate standard deviations from the individual runs and add to summary
        if not ensemble_runs.empty:
            std_df = ensemble_runs.groupby(['cnn_architecture', 'Method'])[ensemble_metrics].agg('std')
            std_df = std_df.reset_index()
            
            # Rename std columns
            std_columns = {}
            for metric in ensemble_metrics:
                std_columns[metric] = f'{metric}_std'
            
            std_df = std_df.rename(columns=std_columns)
            
            # Merge means and stds
            ensemble_summary = pd.merge(
                ensemble_summary, std_df, 
                on=['cnn_architecture', 'Method'], 
                how='left'
            )
        else:
            # If no individual runs are available, add empty std columns
            for metric in ensemble_metrics:
                ensemble_summary[f'{metric}_std'] = None
        
        # Scale by 100 and round to 3 decimal places for consistency with other metrics
        # Keep MCC in original scale (-1 to 1)
        for col in ensemble_summary.columns:
            if col in ensemble_metrics and col != 'MCC':
                ensemble_summary[col] = (ensemble_summary[col] * 100).round(3)
            elif col == 'MCC':
                ensemble_summary[col] = ensemble_summary[col].round(3)
            elif col.endswith('_std'):
                base_metric = col.replace('_std', '')
                if base_metric != 'MCC':
                    ensemble_summary[col] = (ensemble_summary[col] * 100).round(3) if not ensemble_summary[col].isna().all() else ensemble_summary[col]
                else:
                    ensemble_summary[col] = ensemble_summary[col].round(3) if not ensemble_summary[col].isna().all() else ensemble_summary[col]
        
        # Find best ensemble method per CNN architecture
        best_method_per_cnn = ensemble_summary.loc[
            ensemble_summary.groupby('cnn_architecture')['Accuracy'].idxmax()
        ]
        
        # Overall best ensemble method (across all CNN architectures)
        overall_best_method = ensemble_summary.groupby('Method')[ensemble_metrics].mean().reset_index()
        overall_best_method = overall_best_method.sort_values(by='Accuracy', ascending=False)
        
        # Display tables
        if verbose:
            print_summary_table(ensemble_summary, "Ensemble Methods by CNN Architecture")
            print_summary_table(best_method_per_cnn, "Best Ensemble Method per CNN Architecture")
            print_summary_table(overall_best_method, "Overall Performance of Ensemble Methods")
        
        # Save ensemble results
        output_dir = base_path / classifiers_folder
        ensemble_summary.to_csv(output_dir / "Ensemble_Methods_Summary.csv", index=False)
        best_method_per_cnn.to_csv(output_dir / "Best_Ensemble_Method_per_CNN.csv", index=False)
        overall_best_method.to_csv(output_dir / "Overall_Ensemble_Methods.csv", index=False)
        
        # Compare best individual models vs ensemble methods
        # Get the best individual model for each CNN architecture
        best_individual = overall_stats.loc[
            overall_stats.groupby('cnn_architecture')['Accuracy_mean'].idxmax()
        ]
        
        # Create comprehensive comparison dataframe
        # First, identify all unique ensemble methods
        all_ensemble_methods = ensemble_summary['Method'].unique()
        
        # Create a comprehensive comparison that includes all methods with standard deviations
        comparison_columns = ['Architecture', 'Best_Individual_Model', 
                             'Ind_Accuracy', 'Ind_Accuracy_std', 
                             'Ind_MCC', 'Ind_MCC_std',
                             'Ind_Sensitivity', 'Ind_Sensitivity_std',
                             'Ind_Specificity', 'Ind_Specificity_std',
                             'Ind_Precision', 'Ind_Precision_std']
        
        # Add columns for each ensemble method
        for method in all_ensemble_methods:
            # Add both mean and std columns for each ensemble method
            comparison_columns.extend([
                f'{method}_Accuracy', f'{method}_Accuracy_std',
                f'{method}_MCC', f'{method}_MCC_std',
                f'{method}_Sensitivity', f'{method}_Sensitivity_std',
                f'{method}_Specificity', f'{method}_Specificity_std',
                f'{method}_Precision', f'{method}_Precision_std'
            ])
        
        # Add improvement column for the best ensemble method
        comparison_columns.append('Best_Ensemble_Method')
        comparison_columns.append('Best_Ensemble_Accuracy')
        comparison_columns.append('Improvement')
        
        comparison_data = []
        
        for cnn in cnn_architectures:
            try:
                # Get best individual model data
                ind_row = best_individual[best_individual['cnn_architecture'] == cnn].iloc[0]
                ind_model = ind_row['model']
                ind_acc = ind_row['Accuracy_mean']
                ind_mcc = ind_row['MCC_mean']
                
                # Create row with architecture and individual model data including standard deviations
                ind_acc_std = ind_row['Accuracy_std']
                ind_mcc_std = ind_row['MCC_std']
                ind_sens = ind_row['Sensitivity_mean']
                ind_sens_std = ind_row['Sensitivity_std']
                ind_spec = ind_row['Specificity_mean']
                ind_spec_std = ind_row['Specificity_std']
                ind_prec = ind_row['Precision_mean']
                ind_prec_std = ind_row['Precision_std']
                
                row_data = [cnn, ind_model, 
                           ind_acc, ind_acc_std, 
                           ind_mcc, ind_mcc_std,
                           ind_sens, ind_sens_std,
                           ind_spec, ind_spec_std,
                           ind_prec, ind_prec_std]
                
                # Get data for each ensemble method
                best_acc = 0
                best_method = ""
                
                # Filter ensemble summary for this CNN architecture
                cnn_ensemble = ensemble_summary[ensemble_summary['cnn_architecture'] == cnn]
                
                for method in all_ensemble_methods:
                    # Get method data if available
                    method_row = cnn_ensemble[cnn_ensemble['Method'] == method]
                    
                    if not method_row.empty:
                        # Get metrics with standard deviations
                        method_acc = method_row['Accuracy'].values[0]
                        method_acc_std = method_row['Accuracy_std'].values[0] if 'Accuracy_std' in method_row else None
                        
                        method_mcc = method_row['MCC'].values[0]
                        method_mcc_std = method_row['MCC_std'].values[0] if 'MCC_std' in method_row else None
                        
                        method_sens = method_row['Sensitivity'].values[0]
                        method_sens_std = method_row['Sensitivity_std'].values[0] if 'Sensitivity_std' in method_row else None
                        
                        method_spec = method_row['Specificity'].values[0]
                        method_spec_std = method_row['Specificity_std'].values[0] if 'Specificity_std' in method_row else None
                        
                        method_prec = method_row['Precision'].values[0]
                        method_prec_std = method_row['Precision_std'].values[0] if 'Precision_std' in method_row else None
                        
                        # Track best method
                        if method_acc > best_acc:
                            best_acc = method_acc
                            best_method = method
                    else:
                        # Method not available for this CNN
                        method_acc = None
                        method_acc_std = None
                        method_mcc = None
                        method_mcc_std = None
                        method_sens = None
                        method_sens_std = None
                        method_spec = None
                        method_spec_std = None
                        method_prec = None
                        method_prec_std = None
                    
                    # Add all metrics and their standard deviations
                    row_data.extend([
                        method_acc, method_acc_std,
                        method_mcc, method_mcc_std,
                        method_sens, method_sens_std,
                        method_spec, method_spec_std,
                        method_prec, method_prec_std
                    ])
                
                # Add best method and improvement
                improvement = best_acc - ind_acc if best_acc > 0 else None
                row_data.extend([best_method, best_acc, round(improvement, 3) if improvement is not None else None])
                
                comparison_data.append(row_data)
            except (IndexError, KeyError) as e:
                console.print(f"[yellow]Could not create comparison for {cnn}: {str(e)}[/]")
        
        comparison_df = pd.DataFrame(comparison_data, columns=comparison_columns)
        
        # Display and save comparison
        if verbose:
            print_summary_table(comparison_df, "Comparison: Individual Models vs Ensemble Methods")
        
        comparison_df.to_csv(output_dir / "Individual_vs_Ensemble_Comparison.csv", index=False)
        
        # Create visual comparison - update for new comparison_df format
        create_comparison_chart(comparison_df, output_dir, all_ensemble_methods)
        
        console.print("[bold green]Ensemble processing complete![/]")
    else:
        console.print("[bold yellow]No ensemble data was loaded. Skipping ensemble analysis.[/]")
    
    # Display tables for original metrics
    if verbose:
        print_summary_table(summary_df, "Summary by CNN Architecture, Model, and Run")
        print_summary_table(task_analysis, "Analysis by Task, CNN Architecture, and Model")
        print_summary_table(overall_stats, "Overall Statistics by CNN Architecture and Model (All Runs Combined)")
        print_summary_table(best_model_per_task, "Best Model per Task (Considering Accuracy & MCC)")
    
    # 9) Save all tables to CSV in the main folder
    output_dir = base_path / classifiers_folder
    summary_df.to_csv(output_dir / "Summary_by_Model_and_Run.csv", index=False)
    task_analysis.to_csv(output_dir / "Task_Analysis.csv", index=False)
    overall_stats.to_csv(output_dir / "Overall_Statistics.csv", index=False)
    best_model_per_task.to_csv(output_dir / "Best_Model_per_Task.csv", index=False)
    
    # 10) Save overall statistics per CNN architecture (optional, as in original code)
    for cnn in cnn_architectures:
        cnn_stats = overall_stats[overall_stats['cnn_architecture'] == cnn].drop(columns=['cnn_architecture'])
        output_path = base_path / classifiers_folder / Path(cnn) / f"Overall_Statistics_{cnn}.csv"
        cnn_stats.to_csv(output_path, index=False)
    
    # =================== CREATE COMPREHENSIVE COMPARISON ===================
    # If both all methods and ensemble data were processed, create a comprehensive comparison
    if 'all_methods_master_df' in locals() and 'ensemble_master_df' in locals():
        console.print(Panel(Text("Creating Comprehensive Comparison", style="bold green")))
        
        try:
            # Create a comparison of all individual models, all methods, and ensemble methods
            comprehensive_comparison = create_comprehensive_comparison(
                overall_stats, 
                methods_summary if 'methods_summary' in locals() else None,
                ensemble_summary if 'ensemble_summary' in locals() else None,
                cnn_architectures,
                output_dir
            )
            
            if comprehensive_comparison is not None:
                console.print("[bold green]Comprehensive comparison completed![/]")
        except Exception as e:
            console.print(f"[bold red]Error creating comprehensive comparison: {str(e)}[/]")
    
    console.print("[bold green]All processing complete![/]")

def print_summary_table(df, title):
    table = Table(title=f"[bold cyan]{title}[/]")
    for col in df.columns:
        table.add_column(str(col))
    for _, row in df.iterrows():
        table.add_row(*[str(x) for x in row])
    console.print(table)

def create_simple_methods_chart(methods_summary, output_dir):
    """Create simplified charts for methods comparison"""
    # Ensure required columns exist
    required_cols = ['CNN', 'Classifier', 'Method', 'Accuracy_Mean']
    if not all(col in methods_summary.columns for col in required_cols):
        console.print("[yellow]Cannot create methods chart: missing required columns[/]")
        return
    
    # 1. Create chart showing best method per classifier for each CNN
    plt.figure(figsize=(14, 10))
    
    # Get unique CNNs and classifiers
    cnns = methods_summary['CNN'].unique()
    classifiers = methods_summary['Classifier'].unique()
    
    # Find the best method for each CNN and classifier
    best_methods = methods_summary.loc[
        methods_summary.groupby(['CNN', 'Classifier'])['Accuracy_Mean'].idxmax()
    ]
    
    # Set up plotting
    n_cnns = len(cnns)
    n_classifiers = len(classifiers)
    
    fig, axes = plt.subplots(n_cnns, 1, figsize=(12, 5 * n_cnns))
    if n_cnns == 1:
        axes = [axes]  # Make sure axes is a list even if there's only one CNN
    
    # Colors for classifiers
    colors = ['royalblue', 'forestgreen', 'firebrick', 'goldenrod']
    
    for i, cnn in enumerate(cnns):
        cnn_data = best_methods[best_methods['CNN'] == cnn]
        
        # Get sorted data
        classifier_data = []
        for j, clf in enumerate(classifiers):
            clf_data = cnn_data[cnn_data['Classifier'] == clf]
            if not clf_data.empty:
                classifier_data.append({
                    'Classifier': clf,
                    'Method': clf_data['Method'].values[0],
                    'Accuracy': clf_data['Accuracy_Mean'].values[0]
                })
        
        # Sort by accuracy
        classifier_data.sort(key=lambda x: x['Accuracy'], reverse=True)
        
        # Create bar chart
        clf_names = [f"{d['Classifier']} ({d['Method']})" for d in classifier_data]
        accuracies = [d['Accuracy'] for d in classifier_data]
        
        bars = axes[i].bar(clf_names, accuracies, color=colors[:len(classifier_data)])
        
        # Add accuracy values on top of bars
        for bar, acc in zip(bars, accuracies):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{acc:.2f}%', ha='center', va='bottom')
        
        axes[i].set_title(f'Best Method per Classifier for {cnn}')
        axes[i].set_xlabel('Classifier (Method)')
        axes[i].set_ylabel('Accuracy (%)')
        axes[i].set_ylim(0, max(accuracies) + 5)  # Add some space for labels
    
    plt.tight_layout()
    plt.savefig(output_dir / "Best_Methods_by_Classifier.png", dpi=300)
    plt.close()
    
    # 2. Create chart showing best overall method per CNN
    plt.figure(figsize=(12, 8))
    
    # Find the best overall method for each CNN
    best_per_cnn = methods_summary.loc[
        methods_summary.groupby('CNN')['Accuracy_Mean'].idxmax()
    ]
    
    # Sort CNNs by accuracy
    best_per_cnn = best_per_cnn.sort_values('Accuracy_Mean', ascending=False)
    
    # Create labels that include classifier and method
    labels = [f"{row['CNN']}\n{row['Classifier']} ({row['Method']})" 
              for _, row in best_per_cnn.iterrows()]
    
    # Create bar chart
    bars = plt.bar(labels, best_per_cnn['Accuracy_Mean'], color='royalblue')
    
    # Add accuracy values on top of bars
    for bar, acc in zip(bars, best_per_cnn['Accuracy_Mean']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom')
    
    plt.title('Best Overall Method per CNN Architecture')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, max(best_per_cnn['Accuracy_Mean']) + 5)  # Add some space for labels
    plt.tight_layout()
    
    plt.savefig(output_dir / "Best_Overall_Method_per_CNN.png", dpi=300)
    plt.close()
    try:
        # Check if the required columns exist
        if 'cnn_architecture' not in methods_summary.columns or 'model' not in methods_summary.columns:
            console.print("[bold yellow]Cannot create all methods comparison chart: missing required columns[/]")
            return
        
        # Find accuracy column - prefer mean if available
        accuracy_col = None
        for col in ['Accuracy_mean', 'Accuracy']:
            if col in methods_summary.columns:
                accuracy_col = col
                break
        
        if accuracy_col is None:
            console.print("[bold yellow]Cannot create all methods comparison chart: no accuracy column found[/]")
            return
        
        # Check if Method column exists
        if 'Method' not in methods_summary.columns:
            # Simple comparison by model type
            plt.figure(figsize=(12, 8))
            
            # Get unique CNN architectures and models
            architectures = methods_summary['cnn_architecture'].unique()
            models = methods_summary['model'].unique()
            
            # Set up grouped bar chart
            num_architectures = len(architectures)
            num_models = len(models)
            width = 0.8 / num_models
            
            # Plot bars for each model
            colors = ['royalblue', 'forestgreen', 'firebrick', 'goldenrod']
            
            for i, model in enumerate(models):
                model_data = methods_summary[methods_summary['model'] == model]
                # Create dictionary mapping architecture to accuracy for easier plotting
                acc_dict = {row['cnn_architecture']: row[accuracy_col] for _, row in model_data.iterrows()}
                
                # Get values in same order as architectures list
                values = [acc_dict.get(arch, 0) for arch in architectures]
                
                x = [j + width * i for j in range(num_architectures)]
                plt.bar(x, values, width, label=model, color=colors[i % len(colors)])
            
            plt.xlabel('CNN Architecture')
            plt.ylabel(f'{accuracy_col} (%)')
            plt.title('Comparison of Models Across CNN Architectures')
            plt.xticks([i + width * (num_models - 1) / 2 for i in range(num_architectures)], architectures, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            # Save chart
            plt.savefig(output_dir / "Models_Comparison_by_Architecture.png", dpi=300)
            plt.close()
        else:
            # Complex comparison with methods
            # First create a chart comparing methods within each model type
            models = methods_summary['model'].unique()
            
            for model in models:
                model_data = methods_summary[methods_summary['model'] == model]
                methods = model_data['Method'].unique()
                
                if len(methods) <= 1:
                    continue  # Skip if only one method
                
                plt.figure(figsize=(12, 8))
                architectures = model_data['cnn_architecture'].unique()
                
                num_architectures = len(architectures)
                num_methods = len(methods)
                width = 0.8 / num_methods
                
                # Plot bars for each method
                colors = ['royalblue', 'forestgreen', 'firebrick', 'goldenrod', 'purple', 'darkorange', 'teal']
                
                for i, method in enumerate(methods):
                    method_data = model_data[model_data['Method'] == method]
                    # Create dictionary mapping architecture to accuracy for easier plotting
                    acc_dict = {row['cnn_architecture']: row[accuracy_col] for _, row in method_data.iterrows()}
                    
                    # Get values in same order as architectures list
                    values = [acc_dict.get(arch, 0) for arch in architectures]
                    
                    x = [j + width * i for j in range(num_architectures)]
                    plt.bar(x, values, width, label=method, color=colors[i % len(colors)])
                
                plt.xlabel('CNN Architecture')
                plt.ylabel(f'{accuracy_col} (%)')
                plt.title(f'Comparison of {model.upper()} Methods Across CNN Architectures')
                plt.xticks([i + width * (num_methods - 1) / 2 for i in range(num_architectures)], architectures, rotation=45)
                plt.legend()
                plt.tight_layout()
                
                # Save chart
                plt.savefig(output_dir / f"{model}_Methods_Comparison.png", dpi=300)
                plt.close()
            
            # Next create an overall best method per model type chart
            plt.figure(figsize=(14, 8))
            
            # Find best method for each model and CNN architecture
            best_methods = methods_summary.loc[
                methods_summary.groupby(['model', 'cnn_architecture'])[accuracy_col].idxmax()
            ]
            
            # Set up grouped bar chart by model
            models = best_methods['model'].unique()
            architectures = best_methods['cnn_architecture'].unique()
            
            num_architectures = len(architectures)
            num_models = len(models)
            width = 0.8 / num_models
            
            # Plot bars for each model with best method
            colors = ['royalblue', 'forestgreen', 'firebrick', 'goldenrod']
            
            for i, model in enumerate(models):
                model_data = best_methods[best_methods['model'] == model]
                # Create dictionary mapping architecture to accuracy for easier plotting
                acc_dict = {row['cnn_architecture']: row[accuracy_col] for _, row in model_data.iterrows()}
                method_dict = {row['cnn_architecture']: row['Method'] for _, row in model_data.iterrows()}
                
                # Get values in same order as architectures list
                values = [acc_dict.get(arch, 0) for arch in architectures]
                
                x = [j + width * i for j in range(num_architectures)]
                bars = plt.bar(x, values, width, label=model, color=colors[i % len(colors)])
                
                # Add method labels to bars
                for j, (bar, arch) in enumerate(zip(bars, architectures)):
                    if arch in method_dict:
                        method_label = method_dict[arch]
                        plt.text(bar.get_x() + bar.get_width()/2, 5, 
                                method_label, ha='center', rotation=90, 
                                fontsize=8, color='black')
            
            plt.xlabel('CNN Architecture')
            plt.ylabel(f'{accuracy_col} (%)')
            plt.title('Best Method Performance by Model Type and CNN Architecture')
            plt.xticks([i + width * (num_models - 1) / 2 for i in range(num_architectures)], architectures, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            # Save chart
            plt.savefig(output_dir / "Best_Method_by_Model_and_Architecture.png", dpi=300)
            plt.close()
    
    except Exception as e:
        console.print(f"[bold red]Error creating all methods comparison chart: {str(e)}[/]")

def create_comprehensive_comparison(overall_stats, methods_summary, ensemble_summary, cnn_architectures, output_dir):
    """Create a comprehensive comparison of individual models, all methods, and ensemble methods"""
    try:
        # Check if we have all the necessary data
        if overall_stats is None:
            console.print("[bold yellow]Cannot create comprehensive comparison: missing overall stats[/]")
            return None
        
        # Define metrics to compare
        metrics = ['Accuracy', 'MCC', 'Sensitivity', 'Specificity', 'Precision']
        
        # Create a dataframe to store the best results for each category
        columns = ['cnn_architecture', 'best_individual_model', 'best_individual_accuracy', 
                  'best_method_model', 'best_method', 'best_method_accuracy',
                  'best_ensemble_method', 'best_ensemble_accuracy']
        
        comprehensive_data = []
        
        for cnn in cnn_architectures:
            row_data = {'cnn_architecture': cnn}
            
            # Best individual model
            if overall_stats is not None:
                cnn_stats = overall_stats[overall_stats['cnn_architecture'] == cnn]
                if not cnn_stats.empty:
                    best_idx = cnn_stats['Accuracy_mean'].idxmax()
                    row_data['best_individual_model'] = cnn_stats.loc[best_idx, 'model']
                    row_data['best_individual_accuracy'] = cnn_stats.loc[best_idx, 'Accuracy_mean']
            
            # Best method (if available)
            if methods_summary is not None and 'Method' in methods_summary.columns:
                cnn_methods = methods_summary[methods_summary['cnn_architecture'] == cnn]
                if not cnn_methods.empty:
                    # Find accuracy column
                    acc_col = 'Accuracy_mean' if 'Accuracy_mean' in cnn_methods.columns else 'Accuracy'
                    if acc_col in cnn_methods.columns:
                        best_idx = cnn_methods[acc_col].idxmax()
                        row_data['best_method_model'] = cnn_methods.loc[best_idx, 'model']
                        row_data['best_method'] = cnn_methods.loc[best_idx, 'Method']
                        row_data['best_method_accuracy'] = cnn_methods.loc[best_idx, acc_col]
            
            # Best ensemble method
            if ensemble_summary is not None:
                cnn_ensemble = ensemble_summary[ensemble_summary['cnn_architecture'] == cnn]
                if not cnn_ensemble.empty:
                    acc_col = 'Accuracy'  # Ensemble typically uses 'Accuracy'
                    best_idx = cnn_ensemble[acc_col].idxmax()
                    row_data['best_ensemble_method'] = cnn_ensemble.loc[best_idx, 'Method']
                    row_data['best_ensemble_accuracy'] = cnn_ensemble.loc[best_idx, acc_col]
            
            comprehensive_data.append(row_data)
        
        # Create dataframe
        comprehensive_df = pd.DataFrame(comprehensive_data)
        
        # Calculate overall best approach for each CNN architecture
        comprehensive_df['overall_best_category'] = None
        comprehensive_df['overall_best_accuracy'] = 0
        
        categories = [
            ('individual', 'best_individual_accuracy'),
            ('method', 'best_method_accuracy'),
            ('ensemble', 'best_ensemble_accuracy')
        ]
        
        for idx, row in comprehensive_df.iterrows():
            best_category = None
            best_accuracy = 0
            
            for category, acc_col in categories:
                if acc_col in row and pd.notna(row[acc_col]) and row[acc_col] > best_accuracy:
                    best_accuracy = row[acc_col]
                    best_category = category
            
            comprehensive_df.loc[idx, 'overall_best_category'] = best_category
            comprehensive_df.loc[idx, 'overall_best_accuracy'] = best_accuracy
        
        # Save the comprehensive comparison
        comprehensive_df.to_csv(output_dir / "Comprehensive_Comparison.csv", index=False)
        
        # Create visualization
        plt.figure(figsize=(14, 10))
        
        # Set up grouped bar chart
        arch_names = comprehensive_df['cnn_architecture'].tolist()
        ind_acc = comprehensive_df['best_individual_accuracy'].tolist()
        method_acc = comprehensive_df['best_method_accuracy'].tolist() if 'best_method_accuracy' in comprehensive_df.columns else [0] * len(arch_names)
        ensemble_acc = comprehensive_df['best_ensemble_accuracy'].tolist() if 'best_ensemble_accuracy' in comprehensive_df.columns else [0] * len(arch_names)
        
        # Replace None values with 0 for plotting
        method_acc = [acc if pd.notna(acc) else 0 for acc in method_acc]
        ensemble_acc = [acc if pd.notna(acc) else 0 for acc in ensemble_acc]
        
        x = np.arange(len(arch_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        rects1 = ax.bar(x - width, ind_acc, width, label='Best Individual Model', color='royalblue')
        rects2 = ax.bar(x, method_acc, width, label='Best Method', color='forestgreen')
        rects3 = ax.bar(x + width, ensemble_acc, width, label='Best Ensemble', color='firebrick')
        
        # Add labels and title
        ax.set_xlabel('CNN Architecture')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Comprehensive Comparison: Best Approach for Each CNN Architecture')
        ax.set_xticks(x)
        ax.set_xticklabels(arch_names, rotation=45)
        ax.legend()
        
        # Add model/method labels to bars
        for i, rect in enumerate(rects1):
            model = comprehensive_df.iloc[i]['best_individual_model'] if pd.notna(comprehensive_df.iloc[i].get('best_individual_model')) else ''
            ax.text(rect.get_x() + rect.get_width()/2, 5, model, 
                    ha='center', va='bottom', rotation=90, fontsize=8)
        
        for i, rect in enumerate(rects2):
            if 'best_method' in comprehensive_df.columns:
                method = comprehensive_df.iloc[i].get('best_method', '')
                if pd.notna(method):
                    ax.text(rect.get_x() + rect.get_width()/2, 5, method, 
                            ha='center', va='bottom', rotation=90, fontsize=8)
        
        for i, rect in enumerate(rects3):
            if 'best_ensemble_method' in comprehensive_df.columns:
                method = comprehensive_df.iloc[i].get('best_ensemble_method', '')
                if pd.notna(method):
                    ax.text(rect.get_x() + rect.get_width()/2, 5, method, 
                            ha='center', va='bottom', rotation=90, fontsize=8)
        
        # Highlight the overall best for each architecture
        for i, category in enumerate(comprehensive_df['overall_best_category']):
            if category == 'individual':
                rects1[i].set_edgecolor('black')
                rects1[i].set_linewidth(2)
            elif category == 'method':
                rects2[i].set_edgecolor('black')
                rects2[i].set_linewidth(2)
            elif category == 'ensemble':
                rects3[i].set_edgecolor('black')
                rects3[i].set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(output_dir / "Comprehensive_Comparison.png", dpi=300)
        plt.close()
        
        return comprehensive_df
    
    except Exception as e:
        console.print(f"[bold red]Error creating comprehensive comparison: {str(e)}[/]")
        return None

def create_comparison_chart(comparison_df, output_dir, ensemble_methods):
    """Create a bar chart comparing individual models vs ensemble methods"""
    try:
        # Create the main comparison chart
        plt.figure(figsize=(12, 8))
        architectures = comparison_df['Architecture']
        
        x = range(len(architectures))
        
        n_bars = len(ensemble_methods) + 1  # +1 for the individual model
        width = 0.8 / n_bars
        
        # Plot bar for individual model
        plt.bar(x, comparison_df['Ind_Accuracy'], width, label='Best Individual Model', color='royalblue')
        
        # Plot bars for each ensemble method
        colors = ['forestgreen', 'firebrick', 'goldenrod', 'purple', 'darkorange', 'teal']
        
        for i, method in enumerate(ensemble_methods):
            method_col = f'{method}_Accuracy'
            if method_col in comparison_df.columns:
                values = comparison_df[method_col].values
                # Replace None values with 0 for plotting
                values = [v if v is not None else 0 for v in values]
                plt.bar([j + width * (i + 1) for j in x], values, width, 
                        label=f'{method} Ensemble', color=colors[i % len(colors)])
        
        plt.xlabel('CNN Architecture')
        plt.ylabel('Accuracy (%)')
        plt.title('Comparison: Individual Models vs Ensemble Methods')
        plt.xticks([i + width * (n_bars - 1) / 2 for i in x], architectures, rotation=45)
        plt.ylim(0, 110)  # Allow space for text above bars
        
        plt.legend()
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_dir / "Individual_vs_Ensemble_Comparison.png", dpi=300)
        plt.close()
        
        # Create a second chart showing improvement for best ensemble method
        plt.figure(figsize=(10, 6))
        improvements = comparison_df['Improvement'].values
        # Replace None values with 0 for plotting
        improvements = [i if i is not None else 0 for i in improvements]
        
        bars = plt.bar(architectures, improvements, color=['green' if i > 0 else 'red' for i in improvements])
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel('CNN Architecture')
        plt.ylabel('Accuracy Improvement (%)')
        plt.title('Improvement from Best Ensemble Method over Individual Model')
        plt.xticks(rotation=45)
        
        # Add value labels above/below bars
        for bar, improvement in zip(bars, improvements):
            if improvement != 0:
                plt.text(bar.get_x() + bar.get_width()/2, 
                        improvement + (0.5 if improvement > 0 else -1.0),
                        f"{improvement:.1f}", 
                        ha='center')
        
        plt.tight_layout()
        
        # Save chart
        plt.savefig(output_dir / "Ensemble_Improvement.png", dpi=300)
        plt.close()
    except Exception as e:
        console.print(f"[bold red]Error creating comparison chart: {str(e)}[/]")

if __name__ == "__main__":
    main()