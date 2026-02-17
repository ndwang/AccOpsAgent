"""Tests for action plan model."""

import pytest
from accel_agent.core.plan import ActionPlan, ActionType


def test_action_plan_from_dict():
    d = {
        "description": "Test plan",
        "actions": [
            {"type": "read", "params": {"devices": ["BPM:01:X"], "label": "baseline"}},
            {"type": "write", "params": {"device": "COR:01", "value": 1.0, "reason": "bump"}},
        ],
        "rollback": [
            {"type": "write", "params": {"device": "COR:01", "value": 0.0, "reason": "restore"}},
        ],
    }
    plan = ActionPlan.from_dict(d)
    assert plan.description == "Test plan"
    assert len(plan.actions) == 2
    assert plan.actions[0].type == ActionType.READ
    assert plan.actions[1].type == ActionType.WRITE
    assert len(plan.rollback) == 1
