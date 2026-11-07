"""Per-seed score files and their mean and Student-t interval over the paper seeds."""
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from scipy import stats

from .layout import SEEDS


class ScoreError(Exception):
    """A score file or field that a table needs is absent, unreadable or inconsistent across seeds."""


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(value)


@dataclass(frozen=True)
class Stat:
    """Values over seeds, with their mean and two-sided 95% Student-t half-width."""
    values: tuple

    @property
    def mean(self):
        return statistics.mean(self.values)

    @property
    def half_width(self):
        n = len(self.values)
        return stats.t.ppf(0.975, n - 1) * statistics.stdev(self.values) / math.sqrt(n)


class Scores:
    """The score files `<root>/<config>/seed_<n>.json` of the paper seeds."""

    def __init__(self, root):
        self.root = Path(root)
        self._seed_scores = {}

    def path(self, config, seed):
        """Score file of one config and seed."""
        return self.root / config / f"seed_{seed}.json"

    def require(self, configs):
        """Raise ScoreError naming every absent score file of the given configs."""
        missing = [f"  {config} seed {seed}" for config in configs for seed in SEEDS if not self.path(config, seed).is_file()]
        if missing:
            raise ScoreError(f"{len(missing)} score files missing under {self.root}:\n" + "\n".join(missing))

    def seed_scores(self, config):
        """Parsed score files of one config, in seed order."""
        if config not in self._seed_scores:
            self._seed_scores[config] = [self._read(config, seed) for seed in SEEDS]
        return self._seed_scores[config]

    def _read(self, config, seed):
        path = self.path(config, seed)
        if not path.is_file():
            raise ScoreError(f"{config} seed {seed}: missing score file {path}")
        try:
            return json.loads(path.read_text())
        except ValueError as error:
            raise ScoreError(f"{config} seed {seed}: unreadable score file {path} ({error})") from error

    def field(self, config, *keys):
        """Per-seed values of one field, failing on the first seed that lacks it."""
        values = []
        for seed, value in zip(SEEDS, self.seed_scores(config)):
            for key in keys:
                try:
                    value = value[key]
                except (KeyError, IndexError, TypeError):
                    value = None
                if value is None:
                    raise ScoreError(f"{config} seed {seed}: missing {'.'.join(map(str, keys))}")
            values.append(value)
        return values

    def stat(self, config, *keys):
        """Mean and interval of one numeric field over the paper seeds."""
        values = self.field(config, *keys)
        for seed, value in zip(SEEDS, values):
            if not _is_number(value):
                raise ScoreError(f"{config} seed {seed}: {'.'.join(map(str, keys))} is not a number ({value!r})")
        return Stat(tuple(values))

    def shared(self, config, *keys):
        """A field that every seed must hold with the same value."""
        values = self.field(config, *keys)
        for seed, value in zip(SEEDS, values):
            if value != values[0]:
                raise ScoreError(f"{config} seed {seed}: {'.'.join(map(str, keys))} differs from seed {SEEDS[0]}")
        return values[0]
