"""Human-in-the-loop plan review (approval gate)."""

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

from accel_agent.core.plan import ActionPlan, ActionType

console = Console()


class ApprovalGate:
    async def review_plan(self, plan: ActionPlan) -> bool:
        console.print(
            Panel(
                plan.description,
                title="[yellow bold]Proposed Action Plan[/]",
                border_style="yellow",
            )
        )
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", width=4)
        table.add_column("Action", width=8)
        table.add_column("Details")
        for i, action in enumerate(plan.actions):
            table.add_row(
                str(i + 1),
                action.type.value.upper(),
                self._describe_action(action),
            )
        console.print(table)
        if plan.rollback:
            console.print("\n[dim]Rollback on completion/failure:[/]")
            for action in plan.rollback:
                console.print(f"  [dim]← {self._describe_action(action)}[/]")
        console.print()
        has_writes = any(
            a.type in (ActionType.WRITE, ActionType.SCAN) for a in plan.actions
        )
        if not has_writes:
            return True
        write_count = sum(
            1 for a in plan.actions if a.type in (ActionType.WRITE, ActionType.SCAN)
        )
        if write_count == 1:
            return Confirm.ask("[bold]Approve?[/]", default=False)
        choice = Prompt.ask(
            "[bold]Approve?[/]",
            choices=["all", "none", "modify"],
            default="none",
        )
        if choice == "all":
            console.print("[green]✓ Plan approved[/]")
            return True
        if choice == "modify":
            return await self._modify_plan(plan)
        console.print("[red]✗ Plan rejected[/]")
        return False

    async def _modify_plan(self, plan: ActionPlan) -> bool:
        for i, action in enumerate(plan.actions):
            if action.type == ActionType.SCAN:
                console.print(f"\n[cyan]Scan (step {i + 1}):[/]")
                new_steps = Prompt.ask(
                    f"  Steps [{action.params['steps']}]",
                    default=str(action.params["steps"]),
                )
                action.params["steps"] = int(new_steps)
                new_dwell = Prompt.ask(
                    f"  Dwell seconds [{action.params.get('dwell_seconds', 1.0)}]",
                    default=str(action.params.get("dwell_seconds", 1.0)),
                )
                action.params["dwell_seconds"] = float(new_dwell)
            elif action.type == ActionType.WAIT:
                console.print(f"\n[cyan]Wait (step {i + 1}):[/]")
                new_wait = Prompt.ask(
                    f"  Seconds [{action.params['seconds']}]",
                    default=str(action.params["seconds"]),
                )
                action.params["seconds"] = float(new_wait)
        return await self.review_plan(plan)

    def _describe_action(self, action) -> str:
        p = action.params
        match action.type:
            case ActionType.READ:
                devices = ", ".join(p["devices"])
                label = f' ({p["label"]})' if p.get("label") else ""
                return f"{devices}{label}"
            case ActionType.WRITE:
                return f'{p["device"]} → {p["value"]}  ({p.get("reason", "")})'
            case ActionType.WAIT:
                return f'{p["seconds"]}s  ({p.get("reason", "")})'
            case ActionType.SCAN:
                return (
                    f'{p["write_device"]}: {p["start"]} → {p["stop"]} '
                    f'({p["steps"]} steps, {p.get("dwell_seconds", 1.0)}s dwell) '
                    f'reading {", ".join(p["read_devices"])}'
                )
        return ""
