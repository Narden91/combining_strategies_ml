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
    # Scale by 100 and round to 3 decimal places
    for col in summary_df.columns:
        if 'mean' in col or 'std' in col:
            summary_df[col] = (summary_df[col] * 100).round(3)
    
    # 6) Analysis by task, CNN architecture, and model
    task_analysis = master_df.groupby(['cnn_architecture', 'Task', 'model'])[metrics].agg(['mean', 'std'])
    task_analysis.columns = ['_'.join(col) for col in task_analysis.columns]
    task_analysis = task_analysis.sort_values(by=['cnn_architecture', 'model', 'Task']).reset_index()
    # Scale by 100 and round to 3 decimal places
    for col in task_analysis.columns:
        if 'mean' in col or 'std' in col:
            task_analysis[col] = (task_analysis[col] * 100).round(3)
    
    # 7) Overall statistics by CNN architecture and model (all runs combined)
    overall_stats = master_df.groupby(['cnn_architecture', 'model'])[metrics].agg(['mean', 'std'])
    overall_stats.columns = ['_'.join(col) for col in overall_stats.columns]
    overall_stats = overall_stats.reset_index()
    # Scale by 100 and round to 3 decimal places
    for col in overall_stats.columns:
        if 'mean' in col or 'std' in col:
            overall_stats[col] = (overall_stats[col] * 100).round(3)
    
    # 8) Best model for each task (considering both Accuracy and MCC mean)
    task_analysis['score'] = task_analysis['Accuracy_mean'] + task_analysis['MCC_mean']  # Already scaled
    best_model_per_task = task_analysis.loc[task_analysis.groupby('Task')['score'].idxmax()].drop(columns=['score'])
    # Scale score for display consistency (though dropped in final table)
    task_analysis['score'] = task_analysis['score'].round(3)
    
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
        
        # Create summary statistics for ensemble methods
        ensemble_summary = summary_rows.groupby(['cnn_architecture', 'Method'])[ensemble_metrics].agg('mean')
        ensemble_summary = ensemble_summary.reset_index()
        
        # Scale by 100 and round to 3 decimal places for consistency with other metrics
        for col in ensemble_metrics:
            ensemble_summary[col] = (ensemble_summary[col] * 100).round(3)
        
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
        
        # Create a comprehensive comparison that includes all methods
        comparison_columns = ['Architecture', 'Best_Individual_Model', 'Ind_Accuracy', 'Ind_MCC', 'Ind_Sensitivity', 'Ind_Specificity', 'Ind_Precision']
        
        # Add columns for each ensemble method
        for method in all_ensemble_methods:
            comparison_columns.extend([f'{method}_Accuracy', f'{method}_MCC', f'{method}_Sensitivity', f'{method}_Specificity', f'{method}_Precision'])
        
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
                
                # Create row with architecture and individual model data
                ind_sens = ind_row['Sensitivity_mean']
                ind_spec = ind_row['Specificity_mean']
                ind_prec = ind_row['Precision_mean']
                row_data = [cnn, ind_model, ind_acc, ind_mcc, ind_sens, ind_spec, ind_prec]
                
                # Get data for each ensemble method
                best_acc = 0
                best_method = ""
                
                # Filter ensemble summary for this CNN architecture
                cnn_ensemble = ensemble_summary[ensemble_summary['cnn_architecture'] == cnn]
                
                for method in all_ensemble_methods:
                    # Get method data if available
                    method_row = cnn_ensemble[cnn_ensemble['Method'] == method]
                    
                    if not method_row.empty:
                        method_acc = method_row['Accuracy'].values[0]
                        method_mcc = method_row['MCC'].values[0]
                        method_sens = method_row['Sensitivity'].values[0]
                        method_spec = method_row['Specificity'].values[0]
                        method_prec = method_row['Precision'].values[0]
                        
                        # Track best method
                        if method_acc > best_acc:
                            best_acc = method_acc
                            best_method = method
                    else:
                        # Method not available for this CNN
                        method_acc = None
                        method_mcc = None
                        method_sens = None
                        method_spec = None
                        method_prec = None
                    
                    row_data.extend([method_acc, method_mcc, method_sens, method_spec, method_prec])
                
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
    
    console.print("[bold green]All processing complete![/]")

def print_summary_table(df, title):
    table = Table(title=f"[bold cyan]{title}[/]")
    for col in df.columns:
        table.add_column(str(col))
    for _, row in df.iterrows():
        table.add_row(*[str(x) for x in row])
    console.print(table)

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