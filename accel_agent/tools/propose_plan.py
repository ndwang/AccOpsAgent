"""Propose action plans for operator approval (approval-gated)."""

from accel_agent.tools.base import BaseTool
from accel_agent.core.plan import ActionPlan


class ProposePlanTool(BaseTool):
    name = "propose_plan"
    description = """Propose an action plan for operator approval. Use this for
ALL operations that interact with the control system beyond simple reads.

A plan is an ordered sequence of actions. There are four action types:

READ  — Read device values. Use to capture baselines, observe responses,
        or collect data during procedures.
        params: { "devices": ["DEV1", "DEV2"], "label": "optional label" }

WRITE — Set a device to a value. Every write requires operator approval.
        Always capture current values with get_parameter BEFORE proposing
        writes so you can include rollback actions.
        params: { "device": "DEV", "value": 1.23, "reason": "why" }

WAIT  — Pause execution. Use for settling time after writes (magnet ramp,
        RF fill, BPM averaging).
        params: { "seconds": 2.0, "reason": "magnet settling" }

SCAN  — Sweep a device across a range while reading others at each step.
        params: { "write_device": "DEV", "start": -10, "stop": 10,
                  "steps": 20, "read_devices": ["DEV1", "DEV2"],
                  "dwell_seconds": 2.0 }

Compose these to express any procedure. Include rollback actions to restore
original state after the plan completes or if it fails partway through."""

    def __init__(self, approval_gate, plan_executor):
        self.approval = approval_gate
        self.executor = plan_executor

    def parameters_schema(self):
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "Human-readable summary of what this plan does and why.",
                },
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["read", "write", "wait", "scan"],
                            },
                            "params": {"type": "object"},
                        },
                        "required": ["type", "params"],
                    },
                    "description": "Ordered sequence of actions.",
                },
                "rollback": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string", "enum": ["read", "write", "wait"]},
                            "params": {"type": "object"},
                        },
                        "required": ["type", "params"],
                    },
                    "description": "Actions to restore original state after completion or on failure.",
                },
            },
            "required": ["description", "actions"],
        }

    async def execute(self, params):
        plan = ActionPlan.from_dict(params)
        approved = await self.approval.review_plan(plan)
        if not approved:
            return {"status": "rejected", "message": "Operator rejected the plan."}
        result = await self.executor.execute(plan)
        return {"status": "completed", "result": result}
