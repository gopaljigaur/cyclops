"""Declarative tool permissions: allow, ask and deny rules over tool names."""

from fnmatch import fnmatchcase
from typing import Awaitable, Callable, Dict, Iterable, Literal, Optional, Tuple, Union

from cyclops.core.middleware import Deny, ToolCallContext, ToolMiddleware

Permission = Literal["allow", "ask", "deny"]

# Decides an "ask" call: True allows it, False or a Deny blocks it.
ApprovalCallback = Callable[[ToolCallContext], Awaitable[Union[bool, Deny]]]


class ToolPermissions(ToolMiddleware):
    """Allows, denies or asks about each tool call by its tool name.

    Rules are tool names or fnmatch patterns such as "github_*". A call takes
    the strictest rule it matches: deny, then ask, then allow; a call that
    matches none gets `default`. An "ask" call awaits `approve`.
    """

    def __init__(
        self,
        *,
        allow: Iterable[str] = (),
        ask: Iterable[str] = (),
        deny: Iterable[str] = (),
        approve: Optional[ApprovalCallback] = None,
        default: Permission = "allow",
    ) -> None:
        self.rules: Dict[Permission, Tuple[str, ...]] = {
            "deny": tuple(deny),
            "ask": tuple(ask),
            "allow": tuple(allow),
        }
        if approve is None and (self.rules["ask"] or default == "ask"):
            raise ValueError("ask rules need an approve callback")
        self.approve = approve
        self.default = default

    def permission(self, tool_name: str) -> Permission:
        """The permission a call to tool_name gets."""
        for permission, patterns in self.rules.items():
            if any(fnmatchcase(tool_name, p) for p in patterns):
                return permission
        return self.default

    async def before_tool(self, call: ToolCallContext) -> Optional[Deny]:
        permission = self.permission(call.tool_name)
        if permission == "deny":
            return Deny(f"Tool '{call.tool_name}' is not permitted")
        if permission == "ask":
            assert self.approve is not None
            decision = await self.approve(call)
            if isinstance(decision, Deny):
                return decision
            if not decision:
                return Deny(f"Tool '{call.tool_name}' was not approved")
        return None


__all__ = ["ApprovalCallback", "Permission", "ToolPermissions"]
