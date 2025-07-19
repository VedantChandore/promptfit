import typer
from rich import print
from . import optimize_prompt

def main(prompt: str = typer.Argument(..., help="Prompt to optimize."),
         query: str = typer.Argument(..., help="Reference query for relevance scoring."),
         max_tokens: int = typer.Option(2048, help="Token budget for the optimized prompt.")):
    """Optimize a prompt to fit within a token budget."""
    optimized = optimize_prompt(prompt, query, max_tokens=max_tokens)
    print("[bold green]Optimized Prompt:[/bold green]")
    print(optimized)

if __name__ == "__main__":
    typer.run(main) 