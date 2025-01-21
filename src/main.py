import sys

sys.dont_write_bytecode = True

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table
import logging
from pathlib import Path
from utils.helpers import process_data

# Initialize Rich console
console = Console()

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function with Hydra configuration.
    
    Args:
        cfg: Hydra configuration object
    """
    try:
        # Create header
        console.print("=" * 50, style="cyan")
        console.print("🚀 [bold magenta]Application Started[/]", justify="left")
        console.print("=" * 50, style="cyan")
        
        # Create configuration table
        table = Table(show_header=True, header_style="bold cyan", title="Configuration")
        table.add_column("Section", style="dim")
        table.add_column("Parameter", style="magenta")
        table.add_column("Value", style="green")
        
        for section, params in cfg.items():
            for param_name, param_value in params.items():
                table.add_row(
                    section,
                    param_name,
                    str(param_value)
                )
        
        console.print(table)
        console.print()  
        
        # Create directories if they don't exist
        for path_name, path_value in cfg.paths.items():
            Path(path_value).mkdir(parents=True, exist_ok=True)
            console.print(f"📁 Ensuring directory: [cyan]{path_name}[/] -> [green]{path_value}[/]")
        
        console.print() 
        
        # Process data with configuration
        with console.status("[bold yellow]Processing data...[/]") as status:
            result = process_data(
                batch_size=cfg.processing.batch_size,
                num_workers=cfg.processing.num_workers,
                use_gpu=cfg.processing.use_gpu
            )
            console.print("✨ [bold green]Processing complete![/]")
            console.print(f"📊 Processed items: [bold cyan]{result}[/]")
        
        # Final success message
        console.print() 
        console.print("=" * 50, style="cyan")
        console.print("✅ [bold green]Application completed successfully![/]", justify="left")
        console.print("=" * 50, style="cyan")
        
    except Exception as e:
        console.print("\n[bold red]ERROR: An exception occurred[/]")
        console.print_exception()
        sys.exit(1)

if __name__ == "__main__":
    main()