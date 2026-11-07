"""Print the paper's tables."""
import argparse
import sys
from pathlib import Path

from .render import render_tables
from .scores import ScoreError, Scores

REPO_SCORES = Path(__file__).resolve().parents[3] / "scores"


def main(argv=None):
    """Print Tables 1, 2, 3, 8 and 9, or name what is missing on stderr and return 1."""
    parser = argparse.ArgumentParser(prog="python -m ccw_net.tables", description=__doc__)
    parser.add_argument("--scores", type=Path, default=REPO_SCORES,
                        help="directory holding <config>/seed_<n>.json (default: the repository's scores/)")
    args = parser.parse_args(argv)
    if not args.scores.is_dir():
        print(f"failed: no scores directory {args.scores}", file=sys.stderr)
        return 1
    try:
        tables = render_tables(Scores(args.scores))
    except ScoreError as error:
        print(f"failed: {error}", file=sys.stderr)
        return 1
    print("\n\n\n".join(tables))
    return 0


if __name__ == "__main__":
    sys.exit(main())
