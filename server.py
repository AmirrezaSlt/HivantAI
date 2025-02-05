from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import io
import contextlib
import traceback
import uvicorn

app = FastAPI()

class CodeExecutionRequest(BaseModel):
    code: str

class CodeExecutionResponse(BaseModel):
    output: str | None
    error: str | None

def run_user_code(code: str) -> (str, str):
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

    return stdout_buffer.getvalue(), stderr_buffer.getvalue()

@app.post("/execute", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
    # Log the incoming code request.
    print(f"\n=== Incoming Request ===\n{request.code}")
    try:
        stdout_val, stderr_val = run_user_code(request.code)
        response = CodeExecutionResponse(
            output=stdout_val.strip() or None,
            error=stderr_val.strip() or None
        )
        # Log the full execution response.
        print("\n=== Execution Response ===")
        print(response.model_dump())
        return response
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
        print("\n=== Execution Error ===")
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
