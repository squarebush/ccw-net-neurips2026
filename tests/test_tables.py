import re
import shutil

import pytest

TITLES = [
    "Table 1: Aircraft control",
    "Table 2: CarRacing",
    "Table 3: CUB-200",
    "Table 8: Aircraft control, full ablation matrix",
    "Table 9: CarRacing, full ablation matrix",
]


@pytest.fixture(scope="module")
def tables(run_module):
    result = run_module("ccw_net.tables")
    assert result.returncode == 0
    assert result.stderr == ""
    return result.stdout


def row(table, label):
    for line in table.splitlines():
        values = re.split(r"\s{2,}", line.strip())
        if values[0] == label and len(values) > 1:
            return values
    raise AssertionError(f"no row {label!r}")


def test_titles_in_order(tables):
    assert [line for line in tables.splitlines() if re.match(r"^Table \d+:", line)] == TITLES
    assert len(tables.split("\n\n\n")) == len(TITLES)


def test_paper_values(tables):
    t1, t2, t3, t8, t9 = tables.split("\n\n\n")
    assert row(t1, "Lead+Climb")[1] == "62.8* ±16.1"
    assert row(t1, "Lead+Climb")[5] == "94.8 ±2.1"
    assert row(t2, "Straight")[4] == "98.1 ±1.1"
    assert row(t3, "CCW-Net (Ours)")[2] == "91.3 ±0.3"
    assert row(t3, "CBM")[2] == "41.6 ±0.6"
    assert row(t8, "CCW-Net (Ours, 500K samples)")[4] == "97.3 ±1.1"
    assert row(t9, "CCW-Net (Ours, β=1.0)")[3] == "75.3 ±3.8"
    assert row(t8, "CCW-Net (D=16)")[4] == "89.2 ±7.5"
    assert row(t9, "CCW-Net (D=16)")[3] == "63.5 ±9.0"


def test_missing_seed_fails(repo, run_module, tmp_path):
    scores = tmp_path / "scores"
    shutil.copytree(repo / "scores", scores)
    (scores / "cub200" / "cbm" / "seed_7.json").unlink()
    result = run_module("ccw_net.tables", "--scores", scores)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "cub200/cbm seed 7" in result.stderr
