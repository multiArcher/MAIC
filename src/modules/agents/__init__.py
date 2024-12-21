def __getattr__(name):
    if name == "REGISTRY":
        from utils.maker import AgentMaker
        return AgentMaker

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
