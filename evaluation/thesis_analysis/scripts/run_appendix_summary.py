from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

for name in [
    "07_build_appendix_forecast_granularity.py",
    "08_build_appendix_agent_performance.py",
    "09_build_appendix_e2e_performance.py",
    "10_build_appendix_wilcoxon.py",
]:
    print("\n" + "=" * 100)
    print("RUN", name)
    print("=" * 100)

    subprocess.run(
        [sys.executable, str(HERE / name)],
        check=True,
    )
