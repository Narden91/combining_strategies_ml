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
    
    # 1) Define paths
    base_path = Path("C:/Users/Emanuele/Desktop/Smartphone")
    classifiers_folder = Path("MLP")           
    cnn_folder = Path("ConvNextSmall")         
    path_to_classification_ml = base_path / classifiers_folder / cnn_folder / "ClassificationML"
    
    # 2) Define models and number of runs
    models = ['knn', 'nn', 'rf', 'xgb']  # Extend or modify as needed
    total_runs = 20
    
    all_data = []  # Store all dataframes
    
    # 3) Read CSV files with progress tracking
    with Progress() as progress:
        task = progress.add_task("[cyan]Reading CSV Files...", total=len(models) * total_runs)
        
        for model in models:
            for run in range(1, total_runs + 1):
                run_folder = f'run_{run}'
                csv_path = path_to_classification_ml / model / run_folder / 'Metrics.csv'
                
                if csv_path.exists():
                    df = pd.read_csv(csv_path, delimiter=',')
                    df['model'] = model
                    df['run'] = run
                    all_data.append(df)
                else:
                    console.print(f"[bold red]File not found:[/] {csv_path}")
                
                progress.update(task, advance=1)
    
    # 4) Concatenate all data
    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        
        # 5) Remove duplicates
        duplicates = master_df.duplicated(subset=['model', 'run', 'Task'], keep=False)
        if duplicates.any():
            console.print("[bold yellow]Warning: Duplicate rows found![/]")
            master_df = master_df.drop_duplicates(subset=['model', 'run', 'Task'], keep='first')
            console.print("[green]Duplicates removed.[/]")
        
        console.print("[bold green]Data loaded successfully![/]")
    else:
        console.print("[bold red]No data was loaded.[/]")
        return
    
    # 6) Summary statistics per model & run
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'MCC']
    summary_df = master_df.groupby(['model', 'run'])[metrics].agg(['mean', 'std'])
    summary_df.columns = ['_'.join(col).strip() for col in summary_df.columns.values]
    summary_df = summary_df.reset_index()
    
    # 7) Analysis by task
    task_analysis = master_df.groupby(['Task', 'model'])[metrics].agg(['mean', 'std'])
    task_analysis.columns = ['_'.join(col) for col in task_analysis.columns]
    task_analysis = task_analysis.reset_index()
    
    # 8) Overall statistics by model (all runs combined)
    overall_stats = master_df.groupby('model')[metrics].agg(['mean', 'std'])
    overall_stats.columns = ['_'.join(col) for col in overall_stats.columns]
    overall_stats = overall_stats.reset_index()
    
    # 9) Best model for each task (considering both Accuracy and MCC mean)
    task_analysis['score'] = task_analysis['Accuracy_mean'] + task_analysis['MCC_mean']
    best_model_per_task = task_analysis.loc[task_analysis.groupby('Task')['score'].idxmax()].drop(columns=['score'])
    
    # Display tables
    print_summary_table(summary_df, "Summary by Model and Run")
    print_summary_table(task_analysis, "Analysis by Task & Model")
    print_summary_table(overall_stats, "Overall Statistics by Model (All Runs Combined)")
    print_summary_table(best_model_per_task, "Best Model per Task (Considering Accuracy & MCC)")
    
    console.print("[bold green]Processing complete![/]")

def print_summary_table(df, title):
    table = Table(title=f"[bold cyan]{title}[/]")
    
    # Add columns
    for col in df.columns:
        table.add_column(str(col))
    
    # Add rows
    for _, row in df.iterrows():
        table.add_row(*[str(x) for x in row])
    
    console.print(table)
    
if __name__ == "__main__":
    main()