from .point_nav import PointNavConfig, PointNavEnv, DEFAULT_OBSTACLES

__all__ = ["PointNavConfig", "PointNavEnv", "DEFAULT_OBSTACLES"]


def __getattr__(name: str):
    if name in ("MetaWorldReachEnv", "MetaWorldReachConfig"):
        from . import metaworld_reach as _mw

        return getattr(_mw, name)
    if name == "SafetyPointGoalEnv":
        from .safety_point_goal import SafetyPointGoalEnv

        return SafetyPointGoalEnv
    if name == "SafetyPointConfig":
        from .safety_point_goal import SafetyPointConfig

        return SafetyPointConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
