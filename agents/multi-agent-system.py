import sys
from pathlib import Path
from typing import Annotated

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.python_repl_tool import python_repl_tool
from tools.stock_performance_tool import stock_data_tool
from tools.wikipedia_tool import wikipedia_tool