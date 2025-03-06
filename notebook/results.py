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
    classifiers_folder = Path("SimpleCLF")  # or "SimpleCLF", etc.
    cnn_architectures = ["EfficientNetV2S", "ConvNeXtSmall", "ResNet50", "InceptionV3"]  # List of CNN architectures
    models = ['knn', 'nn', 'rf', 'xgb']  # Machine learning models
    total_runs = 20  # Number of runs per model
    
    all_data = []  # To store all dataframes
    
    # Calculate total number of files for progress bar
    total_files = len(cnn_architectures) * len(models) * total_runs
    
    # 2) Read CSV files with progress tracking
    with Progress() as progress:
        task = progress.add_task("[cyan]Reading CSV Files...", total=total_files)
        
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
        
        console.print("[bold green]Data loaded successfully![/]")
    else:
        console.print("[bold red]No data was loaded.[/]")
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
    
    # Display tables
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
    
    console.print("[bold green]Processing complete![/]")

def print_summary_table(df, title):
    table = Table(title=f"[bold cyan]{title}[/]")
    for col in df.columns:
        table.add_column(str(col))
    for _, row in df.iterrows():
        table.add_row(*[str(x) for x in row])
    console.print(table)

if __name__ == "__main__":
    main()