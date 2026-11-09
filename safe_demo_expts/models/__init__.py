from .bc import BCPolicy, act_policy, train_bc

__all__ = ["BCPolicy", "train_bc", "act_policy", "ResidualMLP", "train_residual", "act_with_residual"]


def __getattr__(name: str):
    if name in ("ResidualMLP", "train_residual", "act_with_residual", "save_residual", "load_residual"):
        from . import gated_recovery

        return getattr(gated_recovery, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
