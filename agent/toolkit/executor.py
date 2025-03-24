import io
import contextlib
import traceback
from pydantic import BaseModel
from typing import Optional, Type, Dict, Any
from .tool import BaseTool
from .config import PythonCodeExecutorConfig

class PythonCodeExecutionInput(BaseModel):
    code: str
    timeout: Optional[int] = 5

class PythonCodeExecutionResponse(BaseModel):
    output: str | None
    error: str | None

class PythonCodeExecutor(BaseTool):
    def __init__(self, config: PythonCodeExecutorConfig):
        self.config = config
        super().__init__(self.config.id)

    def _invoke(self, inputs: PythonCodeExecutionInput) -> dict:
        """
        Attempt to mimic REPL behavior:
        - If the provided code is a single expression, evaluate it (using eval)
            and print its repr (if non-None).
        - Otherwise, execute it as a script (using exec).
        Standard output and standard error are captured and returned.
        """
        code = inputs.code
        
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        # Redirect stdout/stderr to capture prints and errors.
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            try:
                # First try compiling the code as an expression.
                compiled_expr = compile(code, "<string>", "eval")
                result = eval(compiled_expr, {})
                if result is not None:
                    print(repr(result))
            except SyntaxError:
                # Not a single expression—treat code as a series of statements.
                try:
                    compiled_code = compile(code, "<string>", "exec")
                    exec(compiled_code, {})
                except Exception:
                    traceback.print_exc()  # Print any errors to stderr
            except Exception:
                traceback.print_exc()  # Print errors from eval to stderr

        response = PythonCodeExecutionResponse(
            output=stdout_buffer.getvalue(),
            error=stderr_buffer.getvalue()
        )
        
        return response.model_dump()

    @property
    def description(self) -> str:
        packages = self.config.python_packages
        package_list = ", ".join(packages) if packages else "no additional packages"
        
        env_vars = [f"* {k}" for k in self.config.environment_variables.keys()]
        env_vars_list = "\n    ".join(env_vars) if env_vars else "none configured"
        
        return f"""
        Executes Python code in a sandboxed environment.

        Environment:
        - Python version: {self.config.python_version}
        - Base image: {self.config.base_image}
        - Available packages: {package_list}
        - System packages: {', '.join(self.config.system_packages) if self.config.system_packages else 'none'}
        - Resource limits: CPU {self.config.resource_limits['cpu']}, Memory {self.config.resource_limits['memory']}
        - Environment variables: {env_vars_list}

        Inputs:
        - code: (required) Python code to execute as a string
        - timeout: (optional, default=5) Maximum execution time in seconds

        Returns:
        A dictionary containing:
        - output: captured stdout from code execution (string or null)
        - error: captured stderr or error messages if any (string or null)

        Usage Tips:
        - Do not use comments in the code
        - Use print() for any data you need in the output
        - Code is executed in a fresh environment each time
        - For expressions (e.g., '2 + 2'), the result will be automatically printed
        - For statements (e.g., 'x = 1'), use print() to see results
        """
    
    @property
    def input_model(self) -> Type[BaseModel]:
        return PythonCodeExecutionInput