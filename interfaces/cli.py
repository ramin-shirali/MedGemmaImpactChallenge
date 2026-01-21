"""
MedGemma Agent Framework - Command Line Interface

Provides interactive CLI for the medical AI agent.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn


class MedGemmaCLI:
    """Command-line interface for MedGemma Agent."""

    def __init__(self):
        self.console = Console()
        self.agent = None
        self.registry = None
        self.history: List[Dict[str, str]] = []

    async def initialize(self, model_id: Optional[str] = None):
        """Initialize the agent and registry."""
        from core.agent import MedGemmaAgent
        from core.registry import ToolRegistry
        from core.config import MedGemmaConfig

        config = MedGemmaConfig()
        if model_id:
            config.model.model_id = model_id

        self.console.print("[bold blue]Initializing MedGemma Agent...[/bold blue]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Loading model and tools...", total=None)

            self.registry = ToolRegistry()
            self.registry.auto_discover()

            self.agent = MedGemmaAgent(config=config)
            await self.agent.initialize()

            progress.remove_task(task)

        tool_count = len(self.registry.list_tools())
        self.console.print(f"[green][OK] Initialized with {tool_count} tools[/green]")

    def print_welcome(self):
        """Print welcome message."""
        welcome_text = """
# MedGemma Agent Framework

A modular medical AI assistant powered by MedGemma.

**Available Commands:**
- `/help` - Show this help message
- `/tools` - List available tools
- `/tool <name>` - Get info about a specific tool
- `/run <tool> <json>` - Run a tool directly
- `/history` - Show conversation history
- `/clear` - Clear conversation history
- `/quit` or `/exit` - Exit the CLI

Type your question or command to get started.
"""
        self.console.print(Panel(Markdown(welcome_text), title="Welcome", border_style="blue"))

    def print_tools(self):
        """Print available tools."""
        if not self.registry:
            self.console.print("[red]Registry not initialized[/red]")
            return

        tool_names = self.registry.list_tools()

        table = Table(title="Available Tools")
        table.add_column("Name", style="cyan")
        table.add_column("Category", style="magenta")
        table.add_column("Description", style="white")
        table.add_column("Version", style="green")

        for tool_name in tool_names:
            try:
                info = self.registry.get_tool_info(tool_name)
                category = info.get("category", "unknown")
                desc = info.get("description", "")
                desc = desc[:50] + "..." if len(desc) > 50 else desc
                table.add_row(
                    info.get("name", tool_name),
                    category,
                    desc,
                    info.get("version", "1.0.0")
                )
            except Exception:
                table.add_row(tool_name, "unknown", "", "1.0.0")

        self.console.print(table)

    def print_tool_info(self, tool_name: str):
        """Print detailed info about a tool."""
        if not self.registry:
            self.console.print("[red]Registry not initialized[/red]")
            return

        try:
            info = self.registry.get_tool_info(tool_name)
        except Exception as e:
            self.console.print(f"[red]Tool '{tool_name}' not found: {e}[/red]")
            return

        panel = Panel(
            f"""**Name:** {info.get('name', tool_name)}
**Description:** {info.get('description', '')}
**Version:** {info.get('version', '1.0.0')}
**Category:** {info.get('category', 'unknown')}

**Input Schema:**
```json
{json.dumps(info.get('input_schema', {}), indent=2)}
```

**Output Schema:**
```json
{json.dumps(info.get('output_schema', {}), indent=2)}
```
""",
            title=f"Tool: {tool_name}",
            border_style="cyan"
        )
        self.console.print(Markdown(panel.renderable))

    async def run_tool(self, tool_name: str, input_json: str):
        """Run a tool directly."""
        if not self.registry:
            self.console.print("[red]Registry not initialized[/red]")
            return

        try:
            tool = await self.registry.get_tool(tool_name)
        except Exception as e:
            self.console.print(f"[red]Tool '{tool_name}' not found: {e}[/red]")
            return

        try:
            input_data = json.loads(input_json)
        except json.JSONDecodeError as e:
            self.console.print(f"[red]Invalid JSON: {e}[/red]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"Running {tool_name}...", total=None)

            try:
                validated_input = tool.validate_input(input_data)
                result = await tool.execute(validated_input)

                progress.remove_task(task)

                if result.success:
                    self.console.print(Panel(
                        json.dumps(result.model_dump(), indent=2, default=str),
                        title="[green]Success[/green]",
                        border_style="green"
                    ))
                else:
                    self.console.print(Panel(
                        json.dumps(result.model_dump(), indent=2, default=str),
                        title="[red]Failed[/red]",
                        border_style="red"
                    ))

            except Exception as e:
                progress.remove_task(task)
                self.console.print(f"[red]Error: {e}[/red]")

    async def process_query(self, query: str):
        """Process a natural language query."""
        if not self.agent:
            self.console.print("[red]Agent not initialized[/red]")
            return

        self.history.append({"role": "user", "content": query})

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Thinking...", total=None)

            try:
                response = await self.agent.process(query)
                progress.remove_task(task)

                self.history.append({"role": "assistant", "content": response})

                self.console.print(Panel(
                    Markdown(response),
                    title="MedGemma",
                    border_style="blue"
                ))

            except Exception as e:
                progress.remove_task(task)
                self.console.print(f"[red]Error: {e}[/red]")

    def print_history(self):
        """Print conversation history."""
        if not self.history:
            self.console.print("[yellow]No conversation history[/yellow]")
            return

        for i, msg in enumerate(self.history):
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                self.console.print(f"[bold cyan]You:[/bold cyan] {content}")
            else:
                self.console.print(f"[bold blue]MedGemma:[/bold blue] {content[:200]}...")
            self.console.print()

    async def run_interactive(self):
        """Run interactive CLI loop."""
        self.print_welcome()

        while True:
            try:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")

                if not user_input.strip():
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=2)
                    command = parts[0].lower()

                    if command in ["/quit", "/exit"]:
                        self.console.print("[yellow]Goodbye![/yellow]")
                        break

                    elif command == "/help":
                        self.print_welcome()

                    elif command == "/tools":
                        self.print_tools()

                    elif command == "/tool" and len(parts) > 1:
                        self.print_tool_info(parts[1])

                    elif command == "/run" and len(parts) > 2:
                        await self.run_tool(parts[1], parts[2])

                    elif command == "/history":
                        self.print_history()

                    elif command == "/clear":
                        self.history = []
                        if self.agent:
                            self.agent.memory.clear()
                        self.console.print("[green]History cleared[/green]")

                    else:
                        self.console.print(f"[red]Unknown command: {command}[/red]")
                        self.console.print("Type /help for available commands")

                else:
                    # Process as natural language query
                    await self.process_query(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Use /quit to exit[/yellow]")
            except EOFError:
                break

    async def run_single(self, query: str, tool: Optional[str] = None, input_json: Optional[str] = None):
        """Run a single query or tool execution."""
        if tool and input_json:
            await self.run_tool(tool, input_json)
        elif query:
            await self.process_query(query)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="MedGemma Agent Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )

    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Single query to process"
    )

    parser.add_argument(
        "-t", "--tool",
        type=str,
        help="Tool to run directly"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="JSON input for tool (use with --tool)"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model ID to use"
    )

    parser.add_argument(
        "--list-tools",
        action="store_true",
        help="List available tools and exit"
    )

    return parser


async def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    cli = MedGemmaCLI()

    # For list-tools, only initialize the registry (no model)
    if args.list_tools:
        from core.registry import ToolRegistry
        cli.registry = ToolRegistry()
        cli.registry.auto_discover()
        cli.print_tools()
        return

    # Full initialization
    await cli.initialize(model_id=args.model)

    if args.interactive or (not args.query and not args.tool):
        # Interactive mode
        await cli.run_interactive()
    else:
        # Single execution mode
        await cli.run_single(
            query=args.query,
            tool=args.tool,
            input_json=args.input
        )


def run():
    """Synchronous entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
