"""Run GROMACS MBAR analysis in a SLURM worker."""

import argparse as _argparse

from ..configuration import EngineType as _EngineType
from .mbar import run_mbar as _run_mbar


def _main() -> None:
    """Parse command-line arguments and run GROMACS MBAR analysis."""
    parser = _argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-no", required=True, type=int)
    parser.add_argument("--percentage-end", required=True, type=float)
    parser.add_argument("--percentage-start", required=True, type=float)
    parser.add_argument("--temperature", required=True, type=float)
    parser.add_argument("--unequilibrated", action="store_true")
    args = parser.parse_args()

    _run_mbar(
        output_dir=args.output_dir,
        run_nos=[args.run_no],
        percentage_end=args.percentage_end,
        percentage_start=args.percentage_start,
        equilibrated=not args.unequilibrated,
        engine_type=_EngineType.GROMACS,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    _main()
