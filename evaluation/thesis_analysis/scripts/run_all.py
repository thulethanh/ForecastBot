from pathlib import Path
import subprocess,sys
HERE=Path(__file__).resolve().parent
for n in ["01_build_overall_summary.py","02_build_router_sql_summary.py","03_build_forecast_summary.py","04_build_explanation_summary.py","05_build_performance_summary.py","06_paired_wilcoxon.py"]:
    print("\n"+"="*100); print("RUN",n); print("="*100)
    subprocess.run([sys.executable,str(HERE/n)],check=True)
