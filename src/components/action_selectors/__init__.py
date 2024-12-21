def __getattr__(name):
    if name == "REGISTRY":
        from utils.maker import ActionSelectorMaker
        return ActionSelectorMaker

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
