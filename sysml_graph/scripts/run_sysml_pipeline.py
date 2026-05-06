import subprocess
import sys
from pathlib import Path

BASE = Path("sysml_graph/scripts")

commands = [
    [sys.executable, str(BASE / "convert_sysml_xmi_to_graph.py")],
    [sys.executable, str(BASE / "validate_sysml_graph.py")],
    [sys.executable, str(BASE / "build_pyg_sysml_graph.py")],
]

for command in commands:
    print("\nRunning:", " ".join(command))
    subprocess.run(command, check=True)
