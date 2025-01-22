"""Console script for rl_salamandra_alignment."""
import rl_salamandra_alignment

import typer
from rich.console import Console

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for rl_salamandra_alignment."""
    console.print("Replace this message by putting your code into "
               "rl_salamandra_alignment.cli.main")
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    


if __name__ == "__main__":
    app()
