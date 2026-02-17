"""Action plan data model (READ, WRITE, WAIT, SCAN)."""

from dataclasses import dataclass, field
from enum import Enum


class ActionType(Enum):
    READ = "read"
    WRITE = "write"
    WAIT = "wait"
    SCAN = "scan"


@dataclass
class Action:
    """A single step in an action plan."""
    type: ActionType
    params: dict


@dataclass
class ActionPlan:
    """
    A complete plan proposed by the agent for operator approval.
    Made of an ordered sequence of actions, a human-readable
    description, and optional rollback actions.
    """
    description: str
    actions: list[Action]
    rollback: list[Action] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ActionPlan":
        return cls(
            description=d["description"],
            actions=[
                Action(type=ActionType(a["type"]), params=a["params"])
                for a in d["actions"]
            ],
            rollback=[
                Action(type=ActionType(a["type"]), params=a["params"])
                for a in d.get("rollback", [])
            ],
        )
