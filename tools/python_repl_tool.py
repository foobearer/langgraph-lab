from langchain_core.tools import tool
from typing import Annotated
from langchain_experimental.utilities import PythonREPL

repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The Python code to execute to generate the chart."]
):
    """
        Use this to execute python code. If you want to see the output of a value, you should
        print it outwith `print(...)`. This is visible to the user. The chart should be displayed
        using `plt.show()`
    """

    try: 
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error {repr(e)}"
    return f"Successfully executed the python REPL tool. \n\n Python code executed: \n\n ---Python Code--- \n\n {code} \n\n ---Code Output--- \n\n {result}"

if __name__ == "__main__":
    code = f"""
import numpy as np

arr = np.arange(0, 9)
print(arr)
print(2 * arr)
"""

    print(python_repl_tool.invoke({"code": code}))