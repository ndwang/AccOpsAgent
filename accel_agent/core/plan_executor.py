"""Executes approved action plans via the control system adapter."""

import asyncio

from accel_agent.core.plan import Action, ActionPlan, ActionType


class PlanExecutor:
    """
    Executes an approved action plan step by step.
    Dispatches each action type to the control system adapter
    and collects results into a structured record.
    """

    def __init__(self, adapter, validator):
        self.adapter = adapter
        self.validator = validator

    async def execute(self, plan: ActionPlan) -> dict:
        results = []

        try:
            for i, action in enumerate(plan.actions):
                result = await self._execute_action(action, step=i)
                results.append(result)
        except Exception as e:
            await self._execute_rollback(plan.rollback)
            return {
                "completed_steps": len(results),
                "total_steps": len(plan.actions),
                "results": results,
                "error": str(e),
                "rolled_back": True,
            }

        if plan.rollback:
            await self._execute_rollback(plan.rollback)

        return {
            "completed_steps": len(results),
            "total_steps": len(plan.actions),
            "results": results,
        }

    async def _execute_action(self, action: Action, step: int) -> dict:
        match action.type:
            case ActionType.READ:
                return await self._do_read(action.params)
            case ActionType.WRITE:
                return await self._do_write(action.params)
            case ActionType.WAIT:
                return await self._do_wait(action.params)
            case ActionType.SCAN:
                return await self._do_scan(action.params)

    async def _do_read(self, params: dict) -> dict:
        readings = {}
        for device in params["devices"]:
            result = await self.adapter.get(device)
            readings[device] = {
                "value": result.value,
                "units": result.units,
                "timestamp": result.timestamp.isoformat(),
                "alarm": result.alarm_status,
            }
        return {
            "action": "read",
            "label": params.get("label"),
            "readings": readings,
        }

    async def _do_write(self, params: dict) -> dict:
        self.validator.check(params["device"], params["value"])
        await self.adapter.set(params["device"], params["value"])
        readback = await self.adapter.get(params["device"])
        return {
            "action": "write",
            "device": params["device"],
            "requested": params["value"],
            "readback": readback.value,
            "success": abs(readback.value - params["value"]) < readback.tolerance,
        }

    async def _do_wait(self, params: dict) -> dict:
        await asyncio.sleep(params["seconds"])
        return {
            "action": "wait",
            "seconds": params["seconds"],
            "reason": params.get("reason", ""),
        }

    async def _do_scan(self, params: dict) -> dict:
        import numpy as np

        setpoints = np.linspace(
            params["start"], params["stop"], params["steps"]
        ).tolist()
        scan_data = []
        for sp in setpoints:
            self.validator.check(params["write_device"], sp)
            await self.adapter.set(params["write_device"], sp)
            await asyncio.sleep(params.get("dwell_seconds", 1.0))
            readings = {"setpoint": sp}
            for dev in params["read_devices"]:
                result = await self.adapter.get(dev)
                readings[dev] = result.value
            scan_data.append(readings)
        return {
            "action": "scan",
            "write_device": params["write_device"],
            "range": [params["start"], params["stop"]],
            "points_completed": len(scan_data),
            "data": scan_data,
        }

    async def _execute_rollback(self, rollback_actions: list[Action]) -> None:
        for action in rollback_actions:
            try:
                await self._execute_action(action, step=-1)
            except Exception:
                pass
