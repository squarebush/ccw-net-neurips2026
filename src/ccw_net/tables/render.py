"""Plain-text layout of the paper's tables."""
from .layout import ABLATION_TABLES, CONFIGS, CUB_ROWS, PER_STATE_TABLES
from .scores import ScoreError

ACTIVE = "*"
MERGED = "§"
GAP = "  "
DCI_KEYS = ("disentanglement", "completeness", "informativeness")
DCI_HEADERS = ("Disentanglement", "Completeness", "Informativeness")
EFFECT_HEADERS = ("All Active ↑", "All Inactive ↓")


def interval(stat, digits, scale=1.0, marker=""):
    """Mean and 95% half-width as 'mean ±half'; the marker follows the mean."""
    return f"{stat.mean * scale:.{digits}f}{marker} ±{stat.half_width * scale:.{digits}f}"


def percent(stat, marker=""):
    """A fraction over seeds as a percentage with one decimal."""
    return interval(stat, 1, 100.0, marker)


def dci(scores, config):
    """DCI disentanglement, completeness and informativeness of one config."""
    return [interval(scores.stat(config, "dci", key), 3) for key in DCI_KEYS]


def text_table(title, spans, header, body, notes=()):
    """Monospace table: spanning headers over column ranges, then rows of values, full-width lines or rules (None)."""
    rows = [header, *(row for row in body if isinstance(row, list))]
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    starts = [sum(widths[:column]) + len(GAP) * column for column in range(len(header))]
    width = starts[-1] + widths[-1]
    rule_width = max([width, *(len(row) for row in body if isinstance(row, str))])

    def row_line(values):
        padded = [value.ljust(widths[0]) if column == 0 else value.rjust(widths[column])
                  for column, value in enumerate(values)]
        return GAP.join(padded).rstrip()

    lines = [title, "=" * rule_width]
    if spans:
        labels, underlines = [" "] * width, [" "] * width
        for first, last, text in spans:
            start, end = starts[first], starts[last] + widths[last]
            offset = start + (end - start - len(text)) // 2
            labels[offset:offset + len(text)] = text
            underlines[start:end] = "-" * (end - start)
        lines += ["".join(labels).rstrip(), "".join(underlines).rstrip()]
    lines += [row_line(header), "-" * rule_width]
    for row in body:
        if row is None:
            lines.append("-" * rule_width)
        else:
            lines.append(row_line(row) if isinstance(row, list) else row)
    lines.append("=" * rule_width)
    return "\n".join([*lines, *notes])


def per_state_table(scores, table):
    """Table 1 or 2: one block per model, a summary line and one row per label combination."""
    header = ["Label combination", *table.slots, *EFFECT_HEADERS]
    body = []
    for name, config in table.blocks:
        slots = scores.shared(config, "slots")
        if tuple(slots) != table.slots:
            raise ScoreError(f"{config}: slots {slots}, expected {list(table.slots)}")
        accuracies = " / ".join(percent(scores.stat(config, key)) for key, _ in table.accuracies)
        summary = (f"  tMSE {interval(scores.stat(config, 'mse'), table.mse_digits)}; concept acc. {accuracies}; "
                   f"DCI {' / '.join(dci(scores, config))}")
        if body:
            body.append(None)
        body += [name, summary]
        for state in table.states:
            active = scores.shared(config, "states", state, "active_slots")
            effects = [percent(scores.stat(config, "states", state, "pm", slot), ACTIVE if slot in active else "")
                       for slot in range(len(table.slots))]
            body.append([state, *effects, percent(scores.stat(config, "states", state, "pm_active")),
                         percent(scores.stat(config, "states", state, "pm_forbidden"))])
    spans = [(1, len(header) - 1, "Concept Effect (%)")]
    return text_table(f"Table {table.number}: {table.title}", spans, header, body,
                      [f"{ACTIVE} active concept"])


def ablation_table(scores, table):
    """Table 8 or 9: one row per model, in groups, with the placebo concept's share below."""
    accuracy_headers = [name for _, name in table.accuracies]
    header = ["Model", "tMSE", *accuracy_headers, *EFFECT_HEADERS, *DCI_HEADERS]
    effects = 2 + len(table.accuracies)
    spans = [(effects, effects + 1, "Concept Effect (%)"), (effects + 2, effects + 4, "DCI")]
    if len(table.accuracies) > 1:
        spans.insert(0, (2, effects - 1, "Concept acc. (%)"))
    body = []
    merged_rows = False
    for group, rows in table.groups:
        if body:
            body.append(None)
        body.append(group)
        for label, config in rows:
            merged = any(scores.field(config, "unscored_states"))
            merged_rows = merged_rows or merged
            body.append([f"{label} {MERGED}" if merged else label,
                         interval(scores.stat(config, "mse"), table.mse_digits),
                         *(percent(scores.stat(config, key)) for key, _ in table.accuracies),
                         percent(scores.stat(config, "overall", "pm_active")),
                         percent(scores.stat(config, "overall", "pm_forbidden")),
                         *dci(scores, config)])
    shares = "; ".join(f"{name} {percent(scores.stat(config, 'overall', 'pm_placebo'))}" for name, config in table.placebo)
    notes = [f"Placebo concept's share of the effect (%): {shares}"]
    if merged_rows:
        notes.append(f"{MERGED} only the merged concept's label combination is scored")
    return text_table(f"Table {table.number}: {table.title}", spans, header, body, notes)


def cub_table(scores):
    """Table 3: concept accuracy, per-input concept effect and DCI on CUB-200."""
    header = ["Model", "Concept acc. (%)", *EFFECT_HEADERS, *DCI_HEADERS]
    spans = [(2, 3, "Concept Effect (%)"), (4, 6, "DCI")]
    body = [[name, percent(scores.stat(config, "acc_ll")), percent(scores.stat(config, "attribution", "pm_active")),
             percent(scores.stat(config, "attribution", "pm_forbidden")), *dci(scores, config)]
            for name, config in CUB_ROWS]
    return text_table("Table 3: CUB-200", spans, header, body)


def render_tables(scores):
    """Tables 1, 2, 3, 8 and 9 as text, after checking that every score file is present."""
    scores.require([config["id"] for config in CONFIGS])
    table_1, table_2 = (per_state_table(scores, table) for table in PER_STATE_TABLES)
    table_8, table_9 = (ablation_table(scores, table) for table in ABLATION_TABLES)
    return [table_1, table_2, cub_table(scores), table_8, table_9]
