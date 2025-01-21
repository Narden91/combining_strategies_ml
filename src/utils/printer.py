from rich.console import Console
from rich.table import Table
from omegaconf import DictConfig
from pathlib import Path

class ConsolePrinter:
    def __init__(self):
        self.console = Console()
        
    def print_header(self, text: str) -> None:
        """Print a formatted header."""
        self.console.print("=" * 50, style="cyan")
        self.console.print(f"🚀 [bold magenta]{text}[/]", justify="left")
        self.console.print("=" * 50, style="cyan")
        
    def print_config_table(self, cfg: DictConfig) -> None:
        """Print configuration as a formatted table."""
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
        
        self.console.print(table)
        self.console.print()
        
    def print_directory_creation(self, path_name: str, path_value: Path) -> None:
        """Print directory creation message."""
        self.console.print(
            f"📁 Ensuring directory: [cyan]{path_name}[/] -> [green]{path_value}[/]"
        )
        
    def print_processing_status(self) -> Console.status:
        """Return a status context for processing."""
        return self.console.status("[bold yellow]Processing data...[/]")
        
    def print_processing_result(self, result: int) -> None:
        """Print processing results."""
        self.console.print("✨ [bold green]Processing complete![/]")
        self.console.print(f"📊 Processed items: [bold cyan]{result}[/]")
        
    def print_footer(self, success: bool = True) -> None:
        """Print footer with success or error message."""
        self.console.print()
        self.console.print("=" * 50, style="cyan")
        if success:
            self.console.print(
                "✅ [bold green]Application completed successfully![/]", 
                justify="left"
            )
        else:
            self.console.print(
                "❌ [bold red]Application failed![/]", 
                justify="left"
            )
        self.console.print("=" * 50, style="cyan")
        
    def print_error(self, error: Exception) -> None:
        """Print error message and stack trace."""
        self.console.print("\n[bold red]ERROR: An exception occurred[/]")
        self.console.print_exception()
    
    def print_spacer(self) -> None:
        """Print an empty line for spacing."""
        self.console.print()
        
    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"⚠️ [bold yellow]Warning: {message}[/]")
        
    def print_info(self, message: str) -> None:
        """Print an information message."""
        self.console.print(f"ℹ️ [bold blue]{message}[/]")

