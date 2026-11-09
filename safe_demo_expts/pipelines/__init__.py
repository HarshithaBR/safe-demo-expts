from .evaluate import evaluate_policy

__all__ = ["evaluate_policy"]


def __getattr__(name: str):
    if name == "evaluate_flexible":
        from .evaluate_flexible import evaluate_flexible

        return evaluate_flexible
    if name == "evaluate_interventions":
        from .evaluate_interventions import evaluate_interventions

        return evaluate_interventions
    if name == "evaluate_safety_point_policy":
        from .evaluate_safety_point import evaluate_policy as evaluate_safety_point_policy

        return evaluate_safety_point_policy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
