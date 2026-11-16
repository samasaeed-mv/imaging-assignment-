"""Re-run the whole project in the right order, on any OS (Windows, macOS, Linux).

    python run_all.py              # Phase 1 -> 2 -> 3 notebooks, group report notebook, end-to-end evaluation
    python run_all.py --slicing    # also re-make the 2D slices from data/raw_volumes first (01_slicing notebooks)

Every notebook is executed in place (its saved outputs are refreshed). A log is written to run_all_log.txt.
The run continues after a failure and lists every failed step at the end (exit code 1 if anything failed)."""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MEMBERS = ["charvi", "sama", "kritikaa", "evangeline"]


def steps(with_slicing):
    out = []
    if with_slicing:
        out += [ROOT / "phase1" / f"01_slicing_{m}.ipynb" for m in MEMBERS]
    out += [ROOT / "phase1" / f"02_enhancement_{m}.ipynb" for m in MEMBERS]      # Phase 1: enhanced slices
    out += [ROOT / "phase2" / f"03_segmentation_{m}.ipynb" for m in MEMBERS]     # Phase 2 needs Phase 1 outputs
    out += [ROOT / "phase3" / f"04_classification_{m}.ipynb" for m in MEMBERS]   # Phase 3 needs Phase 1 + 2 outputs
    out += [ROOT / "report" / "05_report_figures.ipynb",                          # group tables and figures
            ROOT / "evaluate_end_to_end.py"]                                      # raw CT -> final tumour mask
    return out


def run(step, log):
    if step.suffix == ".ipynb":
        cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
               "--ExecutePreprocessor.timeout=1800", str(step)]
    else:
        cmd = [sys.executable, str(step)]
    start = time.time()
    result = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.write(f"==== {step.relative_to(ROOT)}\n{result.stdout}\n")
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"{status:6s} {step.relative_to(ROOT)}  ({time.time() - start:.0f} s)", flush=True)
    return result.returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slicing", action="store_true", help="also re-make the slices from data/raw_volumes")
    args = parser.parse_args()
    failed = []
    with open(ROOT / "run_all_log.txt", "w", encoding="utf-8") as log:
        for step in steps(args.slicing):
            if not run(step, log):
                failed.append(step.relative_to(ROOT))
    print("\nAll steps finished." if not failed else "\nFAILED steps (see run_all_log.txt):\n  " + "\n  ".join(map(str, failed)))
    sys.exit(1 if failed else 0)
