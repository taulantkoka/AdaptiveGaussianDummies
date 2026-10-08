"""Combine OMP, LARS, and knockoff benchmark summaries for the paper.

Each ``--source`` argument has the form ``family=summary.csv``. Use ``omp`` or
``lars`` to retain and tag T-Rex rows, and ``knockoff`` to retain Model-X and
Model-X+ rows. This explicit routing prevents duplicated mixed-method caches
from assigning the wrong forward solver.

The script only reads existing CSV results; it never runs simulations. Pass
``--replot-summary`` to regenerate both figures directly from an already
combined summary CSV without recombining source files.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


TREX_METHODS = (
    "Sphere AG",
    "Spherical T-Rex",
    "T-Rex+DA",
)

KNOCKOFF_METHODS = (
    "Model-X",
    "Model-X+",
)

DISPLAY_METHODS = tuple(
    f"{method} ({solver.upper()})"
    for method in TREX_METHODS
    for solver in ("omp", "lars")
) + KNOCKOFF_METHODS


# Spherical T-Rex is omitted from the main benchmark and retained only in the
# compact direct comparison with Sphere AG.
MAIN_METHODS = tuple(
    method
    for method in DISPLAY_METHODS
    if not method.startswith("Spherical T-Rex")
)

SPHERE_UNIFORM_METHODS = (
    "Sphere AG (OMP)",
    "Sphere AG (LARS)",
    "Spherical T-Rex (OMP)",
    "Spherical T-Rex (LARS)",
)


# Paul Tol's bright qualitative palette.
#
# Model-X and Model-X+ intentionally share one color because they belong to
# the same method family. They are distinguished by line and marker styles.
COLORS = {
    "Sphere AG": "#4477AA",
    "Spherical T-Rex": "#EE6677",
    "T-Rex+DA": "#228833",
    "Model-X": "#AA3377",
    "Model-X+": "#AA3377",
}


# One marker shape per method family. Solver and method variants are encoded
# using fill and line style rather than additional colors.
MARKERS = {
    "Sphere AG": "o",
    "Spherical T-Rex": "^",
    "T-Rex+DA": "D",
    "Model-X": "s",
    "Model-X+": "s",
}


# Offset marker locations for paired methods so that markers at identical
# coordinates do not fully obscure one another.
MARK_EVERY = {
    "Sphere AG (OMP)": (0, 2),
    "Sphere AG (LARS)": (1, 2),
    "Spherical T-Rex (OMP)": (0, 2),
    "Spherical T-Rex (LARS)": (1, 2),
    "T-Rex+DA (OMP)": (0, 2),
    "T-Rex+DA (LARS)": (1, 2),
    "Model-X": (0, 2),
    "Model-X+": (1, 2),
}


STRUCTURE_LABELS = {
    "block_toeplitz": "Block-Toeplitz",
    "block": r"Student-$t_3$ loaded blocks",
    "factor": "Grouped factor",
}


def _read_summary(path):
    """Read a previously aggregated summary CSV."""
    rows = []

    with Path(path).open(newline="") as handle:
        for source in csv.DictReader(handle):
            row = dict(source)

            for key in (
                "x",
                "fdr",
                "fdr_se",
                "tpp",
                "tpp_se",
                "selected_size",
                "selected_size_se",
            ):
                row[key] = float(row[key])

            row["n_valid"] = int(float(row["n_valid"]))
            rows.append(row)

    return rows


def _mean_se(values):
    """Return mean, Monte Carlo standard error, and valid sample count."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if not values.size:
        return np.nan, np.nan, 0

    standard_error = (
        values.std(ddof=1) / np.sqrt(values.size)
        if values.size > 1
        else 0.0
    )

    return (
        float(values.mean()),
        float(standard_error),
        int(values.size),
    )


def _summarize_raw(path, max_rep):
    """Aggregate a raw repetition-level benchmark CSV."""
    grouped = {}

    with Path(path).open(newline="") as handle:
        for source in csv.DictReader(handle):
            rep = int(float(source["rep"]))

            if max_rep is not None and rep >= max_rep:
                continue

            key = (
                source["sweep"],
                source["structure"],
                float(source["x"]),
                source["method"],
            )
            grouped.setdefault(key, []).append(source)

    summary = []

    for (sweep, structure, x_value, method), cells in sorted(
        grouped.items()
    ):
        fdr, fdr_se, n_valid = _mean_se(
            [float(cell["fdp"]) for cell in cells]
        )
        tpp, tpp_se, _ = _mean_se(
            [float(cell["tpp"]) for cell in cells]
        )
        size, size_se, _ = _mean_se(
            [float(cell["selected_size"]) for cell in cells]
        )

        summary.append(
            {
                "sweep": sweep,
                "structure": structure,
                "x": x_value,
                "method": method,
                "fdr": fdr,
                "fdr_se": fdr_se,
                "tpp": tpp,
                "tpp_se": tpp_se,
                "selected_size": size,
                "selected_size_se": size_se,
                "n_valid": n_valid,
            }
        )

    return summary


def _parse_source(value):
    """Parse a command-line source of the form solver=path.csv."""
    try:
        solver, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "sources must have the form solver=summary.csv"
        ) from exc

    solver = solver.lower()

    if solver not in {"omp", "lars", "knockoff"}:
        raise argparse.ArgumentTypeError(
            "solver must be omp, lars, or knockoff"
        )

    return solver, path


def combine(sources, raw_sources=(), max_rep=None):
    """Combine solver-specific summaries without duplicating method rows."""
    combined = {}

    tagged_sources = [
        (solver, path, False)
        for solver, path in sources
    ] + [
        (solver, path, True)
        for solver, path in raw_sources
    ]

    for solver, path, is_raw in tagged_sources:
        source_rows = (
            _summarize_raw(path, max_rep)
            if is_raw
            else _read_summary(path)
        )

        for row in source_rows:
            method = row["method"]

            if method in TREX_METHODS:
                if solver == "knockoff":
                    continue

                display = f"{method} ({solver.upper()})"

            elif method in KNOCKOFF_METHODS:
                if solver != "knockoff":
                    continue

                display = method

            else:
                continue

            row = dict(
                row,
                method=display,
                source=str(path),
            )

            key = (
                row["sweep"],
                row["structure"],
                row["x"],
                display,
            )

            previous = combined.get(key)

            if previous is not None and display in KNOCKOFF_METHODS:
                consistent = (
                    np.isclose(
                        previous["fdr"],
                        row["fdr"],
                        equal_nan=True,
                    )
                    and np.isclose(
                        previous["tpp"],
                        row["tpp"],
                        equal_nan=True,
                    )
                )

                if not consistent:
                    raise ValueError(
                        f"inconsistent duplicate knockoff row for {key}"
                    )

                continue

            combined[key] = row

    return list(combined.values())


def _base_method(display):
    """Remove the solver suffix from a displayed T-Rex method name."""
    if display.endswith(" (OMP)") or display.endswith(" (LARS)"):
        return display.rsplit(" (", 1)[0]

    return display


def _style(display):
    """Return the visual style for a displayed method."""
    base = _base_method(display)

    zorder = {
        "Sphere AG": 6,
        "Spherical T-Rex": 5,
        "T-Rex+DA": 4,
        "Model-X": 3,
        "Model-X+": 2,
    }[base]

    result = {
        "color": COLORS[base],
        "marker": MARKERS[base],
        "markersize": 3.7,
        "markeredgewidth": 0.9,
        "linewidth": 1.4,
        "zorder": zorder,
    }

    if display.endswith(" (OMP)"):
        result.update(
            linestyle="-",
            markerfacecolor=COLORS[base],
        )

    elif display.endswith(" (LARS)"):
        result.update(
            linestyle=(0, (4, 2)),
            markerfacecolor="none",
        )

    elif display == "Model-X":
        result.update(
            linestyle=(0, (6, 2)),
            markerfacecolor=COLORS[base],
        )

    elif display == "Model-X+":
        result.update(
            linestyle=":",
            markerfacecolor="none",
        )

    return result


def _series(rows, sweep, structure, method, metric):
    """Extract one ordered curve and its standard errors."""
    cells = [
        row
        for row in rows
        if row["sweep"] == sweep
        and row["structure"] == structure
        and row["method"] == method
    ]

    cells.sort(key=lambda row: row["x"])

    return (
        np.asarray(
            [row["x"] for row in cells],
            dtype=float,
        ),
        np.asarray(
            [row[metric] for row in cells],
            dtype=float,
        ),
        np.asarray(
            [row[f"{metric}_se"] for row in cells],
            dtype=float,
        ),
    )


def _draw_metric(
    axis,
    rows,
    sweep,
    structure,
    metric,
    methods,
):
    """Draw one TPP or FDR panel without Monte Carlo error bars."""
    panel_upper = 0.0

    anchor_x = {
        row["x"]
        for row in rows
        if row["sweep"] == sweep
        and row["structure"] == structure
        and row["method"] == "Sphere AG (OMP)"
    }

    # Draw Sphere AG last so that its curves remain visible.
    draw_order = tuple(
        method
        for method in methods
        if _base_method(method) != "Sphere AG"
    ) + tuple(
        method
        for method in methods
        if _base_method(method) == "Sphere AG"
    )

    for method in draw_order:
        x, y, error = _series(
            rows,
            sweep,
            structure,
            method,
            metric,
        )

        if anchor_x:
            keep = np.asarray(
                [value in anchor_x for value in x],
                dtype=bool,
            )
            x = x[keep]
            y = y[keep]
            error = error[keep]

        if not x.size:
            continue

        axis.plot(
            x,
            y,
            label=method,
            markevery=MARK_EVERY.get(method, 1),
            **_style(method),
        )

        if metric == "fdr":
            finite_y = y[np.isfinite(y)]

            if finite_y.size:
                panel_upper = max(
                    panel_upper,
                    float(np.max(finite_y)),
                )

    if metric == "fdr":
        axis.axhline(
            0.2,
            color="#555555",
            linestyle=(0, (4, 2)),
            linewidth=1.05,
            zorder=5,
        )

        raw_upper = max(
            0.25,
            1.05 * panel_upper,
        )

        candidates = np.asarray(
            [
                0.25,
                0.30,
                0.40,
                0.50,
                0.60,
                0.70,
                0.80,
                0.90,
                1.00,
            ]
        )

        eligible = candidates[
            candidates >= raw_upper
        ]

        upper = float(
            eligible[0]
            if eligible.size
            else 1.0
        )

        axis.set_ylim(0.0, upper)
        axis.yaxis.set_major_locator(
            MaxNLocator(
                nbins=3,
                min_n_ticks=3,
                steps=[1, 2, 2.5, 5, 10],
            )
        )

    else:
        axis.set_ylim(0.0, 1.02)
        axis.set_yticks(
            [0.0, 0.5, 1.0]
        )

    axis.margins(x=0.035)

    axis.grid(
        color="#E5E5E5",
        linewidth=0.45,
    )

    axis.tick_params(
        axis="both",
        labelsize=6.7,
        length=2.2,
        width=0.6,
    )


def _legend(rows, methods, labels=None):
    """Construct ordered legend handles for the available methods."""
    present = {
        row["method"]
        for row in rows
    }

    ordered = [
        method
        for method in methods
        if method in present
    ]

    handles = [
        Line2D(
            [],
            [],
            label=method,
            **_style(method),
        )
        for method in ordered
    ]

    legend_labels = [
        labels.get(method, method)
        if labels
        else method
        for method in ordered
    ]

    return handles, legend_labels


def _save_figure(figure, path):
    """Save tightly cropped PDF and PNG versions."""
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Resolve the final extents of tick labels, axis labels, and legends
    # before calculating the tight bounding box.
    figure.canvas.draw()

    figure.savefig(
        path,
        bbox_inches="tight",
        pad_inches=0.0,
    )

    figure.savefig(
        path.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.0,
    )

    plt.close(figure)
    
def plot_main(rows, structures, path):
    """Create the full-width benchmark figure."""
    figure = plt.figure(
        figsize=(7.15, 5.20)
    )

    # Layout:
    #   0: SNR heading
    #   1: SNR TPP
    #   2: SNR FDR
    #   3: empty vertical separation
    #   4: rho heading
    #   5: rho TPP
    #   6: rho FDR
    grid = figure.add_gridspec(
        7,
        len(structures),
        height_ratios=(
            0.28,
            1.0,
            1.0,
            0.32,
            0.11,
            1.0,
            1.0,
        ),
        wspace=0.22,
        hspace=0.28,
    )

    snr_heading = figure.add_subplot(
        grid[0, :]
    )
    snr_heading.axis("off")
    snr_heading.text(
        0.5,
        0.62,
        "Signal-to-Noise Ratio Sweep",
        ha="center",
        va="center",
        fontsize=8.3,
        fontweight="semibold",
    )

    # Row 3 is intentionally left empty to separate the two sweep blocks.

    rho_heading = figure.add_subplot(
        grid[4, :]
    )
    rho_heading.axis("off")
    rho_heading.text(
        0.5,
        0.0,
        r"Dependency Parameter Sweep",
        ha="center",
        va="bottom",
        fontsize=8.3,
        fontweight="semibold",
    )

    axes = np.empty(
        (4, len(structures)),
        dtype=object,
    )

    panel_rows = (
        1,
        2,
        5,
        6,
    )

    for row_index, grid_row in enumerate(panel_rows):
        for column in range(len(structures)):
            axes[row_index, column] = figure.add_subplot(
                grid[grid_row, column]
            )

    row_specs = (
        ("snr", "tpp"),
        ("snr", "fdr"),
        ("rho", "tpp"),
        ("rho", "fdr"),
    )

    for column, structure in enumerate(structures):
        for row_index, (sweep, metric) in enumerate(row_specs):
            axis = axes[row_index, column]

            _draw_metric(
                axis,
                rows,
                sweep,
                structure,
                metric,
                MAIN_METHODS,
            )

            axis.set_box_aspect(0.40)

            if row_index == 0:
                axis.set_title(
                    STRUCTURE_LABELS[structure],
                    fontsize=8.2,
                    pad=2,
                )

            # The upper member of each TPP/FDR pair shares its horizontal
            # scale with the lower member.  Show ticks and the axis label only
            # once, on the FDR panel, to reduce clutter in the compact figure.
            if row_index in (0, 2):
                axis.tick_params(axis="x", labelbottom=False)

            if row_index == 1:
                axis.set_xlabel(
                    "SNR",
                    fontsize=7.3,
                    labelpad=2,
                )

            elif row_index == 3:
                axis.set_xlabel(
                    r"$\rho$",
                    fontsize=7.3,
                    labelpad=2,
                )

            if column == 0:
                axis.set_ylabel(
                    "FDR" if metric == "fdr" else "TPP",
                    fontsize=7.3,
                    labelpad=3,
                )

    handles, labels = _legend(
        rows,
        MAIN_METHODS,
    )

    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=6.8,
        handlelength=2.5,
        columnspacing=1.0,
        handletextpad=0.45,
        bbox_to_anchor=(0.5, 0.995),
    )

    figure.subplots_adjust(
        left=0.075,
        right=0.995,
        bottom=0.065,
        top=0.89,
    )

    _save_figure(
        figure,
        path,
    )

def plot_sphere_uniform(
    rows,
    path,
    structure="block_toeplitz",
):
    """Compare Sphere AG and spherical T-Rex under grouped dependence."""
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(2.5, 1.5),
        sharex=True,
        squeeze=False,
    )

    axes = axes[:, 0]

    comparison_labels = {
        "Sphere AG (OMP)": "Sphere AG (OMP)",
        "Sphere AG (LARS)": "Sphere AG (LARS)",
        "Spherical T-Rex (OMP)": "Spherical T-Rex (OMP)",
        "Spherical T-Rex (LARS)": "Spherical T-Rex (LARS)",
    }

    for axis, metric in zip(
        axes,
        ("tpp", "fdr"),
    ):
        _draw_metric(
            axis,
            rows,
            sweep="rho",
            structure=structure,
            metric=metric,
            methods=SPHERE_UNIFORM_METHODS,
        )

        axis.set_ylabel(
            "TPP"
            if metric == "tpp"
            else "FDR",
            fontsize=7.3,
            labelpad=3,
        )

    # axes[0].set_title(
    #     STRUCTURE_LABELS[structure]
    #     + r": $\rho$ sweep",
    #     fontsize=8.2,
    #     pad=4,
    # )

    axes[-1].set_xlabel(
        r"$\rho$",
        fontsize=7.3,
        labelpad=2,
    )

    handles, labels = _legend(
        rows,
        SPHERE_UNIFORM_METHODS,
        comparison_labels,
    )

    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=6.3,
        handlelength=2.3,
        columnspacing=0.9,
        handletextpad=0.4,
        bbox_to_anchor=(0.5, 0.995),
    )

    figure.subplots_adjust(
        left=0.17,
        right=0.98,
        bottom=0.14,
        top=0.77,
        hspace=0.16,
    )
    
    _save_figure(
        figure,
        path,
    )


def write_summary(rows, path):
    """Write the combined summary CSV."""
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = (
        "sweep",
        "structure",
        "x",
        "method",
        "fdr",
        "fdr_se",
        "tpp",
        "tpp_se",
        "selected_size",
        "selected_size_se",
        "n_valid",
        "source",
    )

    ordered = sorted(
        rows,
        key=lambda row: (
            row["sweep"],
            row["structure"],
            row["x"],
            row["method"],
        ),
    )

    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )
        writer.writeheader()

        writer.writerows(
            {
                field: row[field]
                for field in fields
            }
            for row in ordered
        )


def _parser():
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--source",
        action="append",
        type=_parse_source,
        default=[],
        help=(
            "solver=summary.csv; repeat "
            "for every component summary"
        ),
    )

    parser.add_argument(
        "--raw-source",
        action="append",
        type=_parse_source,
        default=[],
        help=(
            "solver=raw.csv; repeat as "
            "needed and optionally "
            "restrict repetitions"
        ),
    )

    parser.add_argument(
        "--max-rep",
        type=int,
        default=None,
        help=(
            "For raw sources, retain "
            "repetitions with rep < MAX_REP."
        ),
    )

    parser.add_argument(
        "--replot-summary",
        default=None,
        help=(
            "Replot from an existing combined summary CSV. "
            "When supplied, --source, --raw-source, and "
            "--max-rep are not used."
        ),
    )

    parser.add_argument(
        "--structures",
        nargs="+",
        choices=list(STRUCTURE_LABELS),
        default=[
            "block_toeplitz",
            "block",
            "factor",
        ],
    )

    parser.add_argument(
        "--plot",
        default=(
            "ag_icassp/figs/"
            "final_selector_comparison.pdf"
        ),
    )

    parser.add_argument(
        "--comparison-plot",
        default=(
            "ag_icassp/figs/"
            "sphere_uniform_comparison.pdf"
        ),
    )

    parser.add_argument(
        "--summary",
        default=(
            "ag_icassp/results/"
            "final_selector_comparison.summary.csv"
        ),
    )

    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)

    if args.replot_summary is not None:
        if (
            args.source
            or args.raw_source
            or args.max_rep is not None
        ):
            parser.error(
                "--replot-summary cannot be combined with "
                "--source, --raw-source, or --max-rep"
            )

        all_rows = _read_summary(
            args.replot_summary
        )

        print(
            f"read {args.replot_summary}"
        )

    else:
        if (
            not args.source
            and not args.raw_source
        ):
            parser.error(
                "provide at least one --source or --raw-source, "
                "or use --replot-summary"
            )

        all_rows = combine(
            args.source,
            args.raw_source,
            args.max_rep,
        )

        write_summary(
            all_rows,
            args.summary,
        )

        print(
            f"wrote {args.summary}"
        )

    main_rows = [
        row
        for row in all_rows
        if row["structure"] in args.structures
    ]

    plot_main(
        main_rows,
        args.structures,
        args.plot,
    )

    # The compact comparison always uses the grouped-factor rho sweep,
    # regardless of which structures are displayed in the main figure.
    plot_sphere_uniform(
        all_rows,
        args.comparison_plot,
        structure="factor",
    )

    print(
        f"wrote {args.plot}"
    )
    print(
        f"wrote {args.comparison_plot}"
    )

    for method in DISPLAY_METHODS:
        counts = [
            row["n_valid"]
            for row in all_rows
            if row["method"] == method
        ]

        if counts:
            print(
                f"{method:28s}: "
                f"n={min(counts)}..{max(counts)}"
            )


if __name__ == "__main__":
    main()
