from .obstacle_shield import ObstacleShield, ShieldResult, margin_pair

__all__ = ["ObstacleShield", "ShieldResult", "margin_pair"]


def __getattr__(name: str):
    if name == "SafetyPointShield":
        from .safety_point_shield import SafetyPointShield

        return SafetyPointShield
    if name == "MPCEscapeShield":
        from .mpc_escape import MPCEscapeShield

        return MPCEscapeShield
    if name == "SafetyShieldResult":
        from .safety_point_shield import ShieldResult as SafetyShieldResult

        return SafetyShieldResult
    if name == "TabletopShield":
        from .tabletop_shield import TabletopShield

        return TabletopShield
    if name == "make_tabletop_shield":
        from .tabletop_shield import make_tabletop_shield

        return make_tabletop_shield
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
