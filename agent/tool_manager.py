from .config import Config, ToolConfig

class ToolManager:
    def __init__(self, config: Config):
        self.config = config.TOOL_MANAGER
        self.tools = {
            tool_name: tool_config.TOOL_CLASS(**tool_config.TOOL_KWARGS)
            for tool_name, tool_config in self.config.TOOLS.items()
        }
