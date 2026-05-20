"""Entry point: запускает Streamlit dashboard."""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

subprocess.run([
    sys.executable, "-m", "streamlit", "run",
    "dashboard/app.py",
    "--server.port", "8501",
    "--server.headless", "false",
    "--theme.base", "dark",
])
