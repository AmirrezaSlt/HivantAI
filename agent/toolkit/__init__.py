from .config import ToolkitConfig

class Toolkit:
    def __init__(self, *args, **kwargs):
        self.config = ToolkitConfig(*args, **kwargs)
        self.tools = { tool.id: tool for tool in self.config.TOOLS }
