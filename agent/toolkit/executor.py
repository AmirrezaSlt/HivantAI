import io
import contextlib
import traceback
from pydantic import BaseModel
from typing import Optional, Type, Dict, Any
from .tool import ToolInfo, BaseTool
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

    def execute(self, code, timeout: Optional[int] = 5) -> dict:
        """
        Attempt to mimic REPL behavior:
        - If the provided code is a single expression, evaluate it (using eval)
            and print its repr (if non-None).
        - Otherwise, execute it as a script (using exec).
        Standard output and standard error are captured and returned.
        """
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
        """
        Returns a description of the executor's capabilities as a prompt
        to inform the model about available code execution functionality.
        """
        packages = self.config.python_packages
        package_list = ", ".join(packages) if packages else "no additional packages"
        
        env_vars = [f"* {k}" for k in self.config.environment_variables.keys()]
        env_vars_list = "\n    ".join(env_vars) if env_vars else "none configured"
        
        return f"""
            Python Code Execution Environment:
            - Python version: {self.config.python_version}
            - Base image: {self.config.base_image}
            - Python packages: {package_list}
            - System packages: {', '.join(self.config.system_packages) if self.config.system_packages else 'none'}
            - Resource limits: CPU {self.config.resource_limits['cpu']}, Memory {self.config.resource_limits['memory']}
            - Environment variables: {env_vars_list}

            Capabilities:
            - Executes Python code snippets
            - Captures stdout/stderr
            - Do not use any comments
            - print the data you need to have it in the output
            """
    
    def _invoke(self, input_data: PythonCodeExecutionInput) -> dict:
        """
        Implementation of the abstract _invoke method required by BaseTool.
        Delegates execution to the execute method.
        """
        return self.execute(input_data.code, input_data.timeout)
        
    @property
    def input_model(self) -> Type[BaseModel]:
        return PythonCodeExecutionInput
            
    @property
    def input_schema(self) -> str:
        """
        Returns a description of the expected input format for code execution.
        """
        return PythonCodeExecutionInput.model_json_schema()
            
    @property
    def output_schema(self) -> str:
        """
        Returns a description of the output format from code execution.
        """
        return PythonCodeExecutionResponse.model_json_schema()
    
    @property
    def info(self) -> ToolInfo:
        return ToolInfo(
            description=self.description,
            inputs=self.input_schema,
            outputs=self.output_schema
        )

    def __dict__(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema
        }