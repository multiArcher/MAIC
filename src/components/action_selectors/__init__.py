def __getattr__(name):
    """For compatibility with original implementation of PyMARL."""
    if name == "REGISTRY":
        from utils.maker import ActionSelectorMaker
        return ActionSelectorMaker

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
