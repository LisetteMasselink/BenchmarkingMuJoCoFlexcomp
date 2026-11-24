"""Analysis of saved simulation results, separate from the simulation itself:

    python src/analysis.py output/case1_SVK_E55000_xmlmesh_beta1e-05/case1_results.json [more json ...]
    python src/analysis.py --case case1            (every saved case1 run in output/)

Ramp runs (cases 1-3): comparison with COMSOL (validation_analysis.py), figures in <run folder>/analysis.
Sine runs (case 4): per-cycle analysis (dynamic_analysis.py), all given runs compared in output/<case>_analysis.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import CONFIG
from utils import load_data


def analyse_ramp(results_path):
    """MuJoCo vs COMSOL for one ramp run; figures next to the results."""
    from validation_analysis import analyse as compare
    _, meta = load_data(results_path)
    comsol_case = meta.get("comsol") or meta["case"]
    outdir = Path(results_path).parent / "analysis"
    compare(comsol_case, str(results_path), str(outdir))
    print(f"analysis written to {outdir}")

def analyse(results_paths):
    """Pick the analysis per run from its metadata (protocol ramp or sine)."""
    print(results_paths)
    for path in results_paths:
        analyse_ramp(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyse saved MuJoCo flexcomp validation results.")
    parser.add_argument("results", nargs="*", help="results json file(s) written by a simulation run (--save_data)")
    parser.add_argument("--case", choices=["case1", "case2", "case3", "case4"],
                        help="analyse every saved run of this case in the output folder")
    parser.add_argument("--show", action="store_true", help="show the figures after saving them")
    args = parser.parse_args(argv)
    if args.case:
        args.results += [str(p) for p in sorted(CONFIG.OUTPUT_DIR.glob(f"{args.case}_*/{args.case}_results.json"))]
    if not args.results:
        parser.error(f"no results found: give results json file(s) or --case "
                     f"(runs are saved in {CONFIG.OUTPUT_DIR} with --save_data)")
    return args


def main(argv=None):
    args = parse_args(argv)
    analyse(args.results)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
