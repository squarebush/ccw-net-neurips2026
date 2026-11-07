"""The rows of the paper's Tables 1, 2, 3, 8 and 9 and the configurations behind them."""
from dataclasses import dataclass

SEEDS = tuple(range(10))


@dataclass(frozen=True)
class PerStateTable:
    """Concept effects per label combination, one block of rows per model (Tables 1 and 2)."""
    number: int
    title: str
    slots: tuple
    states: tuple
    accuracies: tuple
    mse_digits: int
    blocks: tuple


@dataclass(frozen=True)
class AblationTable:
    """One row per model in titled groups, with the placebo concept's share as a note (Tables 8 and 9)."""
    number: int
    title: str
    accuracies: tuple
    mse_digits: int
    groups: tuple
    placebo: tuple


AIRCRAFT_ACCURACIES = (("acc_ll", "Lead/Lag"), ("acc_cd", "Climb/Dive"))
CARRACING_ACCURACIES = (("acc_ll", "Concept acc. (%)"),)

TABLE_1 = PerStateTable(
    number=1,
    title="Aircraft control",
    slots=("Lead", "Lag", "Climb", "Dive"),
    states=("Lead+Climb", "Lead+Dive", "Lag+Climb", "Lag+Dive"),
    accuracies=AIRCRAFT_ACCURACIES,
    mse_digits=3,
    blocks=(
        ("CCW-Net (Ours)", "aircraft/width64/ccwnet"),
        ("CCW-Net, no causal alignment losses", "aircraft/width64/noalign"),
        ("CW-Net (Kenny et al.)", "aircraft/cwnet/scalar"),
    ),
)

TABLE_2 = PerStateTable(
    number=2,
    title="CarRacing",
    slots=("Straight", "Left", "Right"),
    states=("Straight", "Left", "Right"),
    accuracies=CARRACING_ACCURACIES,
    mse_digits=4,
    blocks=(
        ("CCW-Net (Ours)", "carracing/width64/ccwnet_d1"),
        ("CCW-Net, no causal alignment losses", "carracing/width64/noalign_d1"),
        ("CW-Net (Kenny et al.)", "carracing/cwnet/scalar"),
    ),
)

CUB_ROWS = (("CCW-Net (Ours)", "cub200/ccwnet"), ("CBM", "cub200/cbm"))

TABLE_8 = AblationTable(
    number=8,
    title="Aircraft control, full ablation matrix",
    accuracies=AIRCRAFT_ACCURACIES,
    mse_digits=3,
    groups=(
        ("Ours", (
            ("CCW-Net (Ours, 500K samples)", "aircraft/width256/ccwnet"),
            ("CCW-Net (Ours, 5M samples)", "aircraft/width256/ccwnet_5m"),
            ("CCW-Net (Ours, scalar D=1)", "aircraft/width256/ccwnet_d1"),
        )),
        ("No causal alignment losses", (
            ("CCW-Net (D=16)", "aircraft/width256/noalign"),
            ("CCW-Net (scalar D=1)", "aircraft/width256/noalign_d1"),
        )),
        ("Architecture baselines", (
            ("CW-Net architecture, scalar + alignment", "aircraft/cwnet/scalar_aligned"),
            ("CW-Net (Kenny et al.)", "aircraft/cwnet/scalar"),
        )),
        ("Disentanglement baselines", (
            ("β-TCVAE (D=16)", "aircraft/disentanglement/btcvae_d16"),
            ("iVAE (D=16)", "aircraft/disentanglement/ivae_d16"),
        )),
        ("Data quantity", (
            ("5K samples", "aircraft/width256/data_5k"),
            ("50K samples", "aircraft/width256/data_50k"),
        )),
        ("Label fraction", (
            ("10% labels", "aircraft/width256/labels_10"),
            ("25% labels", "aircraft/width256/labels_25"),
            ("50% labels", "aircraft/width256/labels_50"),
        )),
        ("Label noise", (
            ("10% noise", "aircraft/width256/noise_10"),
            ("20% noise", "aircraft/width256/noise_20"),
        )),
        ("Loss combinations (without P)", (
            ("N", "aircraft/width256/losses_n"),
            ("S", "aircraft/width256/losses_s"),
            ("I", "aircraft/width256/losses_i"),
            ("N+S", "aircraft/width256/losses_ns"),
            ("N+I", "aircraft/width256/losses_ni"),
            ("S+I", "aircraft/width256/losses_si"),
            ("N+S+I", "aircraft/width256/losses_nsi"),
        )),
        ("Loss combinations (with P)", (
            ("P", "aircraft/width256/losses_p"),
            ("N+P", "aircraft/width256/losses_np"),
            ("S+P", "aircraft/width256/losses_sp"),
            ("I+P", "aircraft/width256/losses_ip"),
            ("N+S+P", "aircraft/width256/losses_nsp"),
            ("N+I+P", "aircraft/width256/losses_nip"),
        )),
        ("Concept ontology, merged concepts", (
            ("Merge Climb+Dive", "aircraft/width256/merge_climb_dive"),
            ("Merge Lead+Lag", "aircraft/width256/merge_lead_lag"),
        )),
        ("Concept ontology, removed concepts", (
            ("Remove Climb", "aircraft/width256/remove_climb"),
            ("Remove Dive", "aircraft/width256/remove_dive"),
            ("Remove Lag", "aircraft/width256/remove_lag"),
            ("Remove Lead", "aircraft/width256/remove_lead"),
        )),
        ("Placebo", (
            ("Placebo concept added", "aircraft/width256/placebo"),
        )),
        ("Encoder width 64", (
            ("CCW-Net (Ours), D=16", "aircraft/width64/ccwnet"),
            ("CCW-Net (Ours), D=8", "aircraft/width64/ccwnet_d8"),
            ("10% label noise", "aircraft/width64/noise_10"),
            ("10% label noise, no alignment", "aircraft/width64/noise_10_noalign"),
            ("Placebo concept added", "aircraft/width64/placebo"),
            ("Placebo concept added, no alignment", "aircraft/width64/placebo_noalign"),
        )),
    ),
    placebo=(
        ("width 256", "aircraft/width256/placebo"),
        ("width 64", "aircraft/width64/placebo"),
        ("width 64, no alignment", "aircraft/width64/placebo_noalign"),
    ),
)

TABLE_9 = AblationTable(
    number=9,
    title="CarRacing, full ablation matrix",
    accuracies=CARRACING_ACCURACIES,
    mse_digits=4,
    groups=(
        ("Ours", (
            ("CCW-Net (Ours, β=1.0)", "carracing/width256/ccwnet"),
            ("CCW-Net (Ours, scalar D=1)", "carracing/width256/ccwnet_d1"),
        )),
        ("No causal alignment losses", (
            ("CCW-Net (D=16)", "carracing/width256/losses_none"),
            ("CCW-Net (scalar D=1)", "carracing/width256/noalign_d1"),
        )),
        ("Architecture baselines", (
            ("CW-Net architecture, scalar + alignment", "carracing/cwnet/scalar_aligned"),
            ("CW-Net (Kenny et al.)", "carracing/cwnet/scalar"),
            ("CW-Net architecture, D=16 + alignment", "carracing/cwnet/vector_aligned"),
            ("CW-Net architecture, D=16", "carracing/cwnet/vector"),
        )),
        ("Disentanglement baselines", (
            ("β-TCVAE (D=1)", "carracing/disentanglement/btcvae_d1"),
            ("β-TCVAE (D=16)", "carracing/disentanglement/btcvae_d16"),
            ("iVAE (D=1)", "carracing/disentanglement/ivae_d1"),
            ("iVAE (D=16)", "carracing/disentanglement/ivae_d16"),
        )),
        ("Label fraction", (
            ("10% labels", "carracing/width256/labels_10"),
            ("25% labels", "carracing/width256/labels_25"),
            ("50% labels", "carracing/width256/labels_50"),
        )),
        ("Label noise", (
            ("10% noise", "carracing/width256/noise_10"),
            ("20% noise", "carracing/width256/noise_20"),
        )),
        ("Loss combinations (without P)", (
            ("N", "carracing/width256/losses_n"),
            ("S", "carracing/width256/losses_s"),
            ("I", "carracing/width256/losses_i"),
            ("N+S", "carracing/width256/losses_ns"),
            ("N+I", "carracing/width256/losses_ni"),
            ("S+I", "carracing/width256/losses_si"),
            ("N+S+I", "carracing/width256/losses_nsi"),
        )),
        ("Loss combinations (with P)", (
            ("P", "carracing/width256/losses_p"),
            ("N+P", "carracing/width256/losses_np"),
            ("S+P", "carracing/width256/losses_sp"),
            ("I+P", "carracing/width256/losses_ip"),
            ("N+S+P", "carracing/width256/losses_nsp"),
            ("N+I+P", "carracing/width256/losses_nip"),
            ("S+I+P", "carracing/width256/losses_sip"),
        )),
        ("Concept ontology, merged concepts", (
            ("Merge Left+Right", "carracing/width256/merge_left_right"),
            ("Merge Straight+Left", "carracing/width256/merge_straight_left"),
            ("Merge Straight+Right", "carracing/width256/merge_straight_right"),
        )),
        ("Concept ontology, removed concepts", (
            ("Remove Left", "carracing/width256/remove_left"),
            ("Remove Right", "carracing/width256/remove_right"),
            ("Remove Straight", "carracing/width256/remove_straight"),
        )),
        ("Placebo", (
            ("Placebo concept added", "carracing/width256/placebo"),
        )),
        ("Encoder width 64", (
            ("CCW-Net (Ours), D=16", "carracing/width64/ccwnet"),
            ("CCW-Net (Ours), D=8", "carracing/width64/ccwnet_d8"),
        )),
    ),
    placebo=(
        ("width 256", "carracing/width256/placebo"),
    ),
)

PER_STATE_TABLES = (TABLE_1, TABLE_2)
ABLATION_TABLES = (TABLE_8, TABLE_9)


def _paper_configs():
    configs = [config for table in PER_STATE_TABLES for _, config in table.blocks]
    configs += [config for table in ABLATION_TABLES for _, rows in table.groups for _, config in rows]
    configs += [config for _, config in CUB_ROWS]
    return list(dict.fromkeys(configs))


CONFIGS = [{"id": config} for config in _paper_configs()]
