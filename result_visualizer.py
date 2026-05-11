"""
result_visualizer.py

Visualization utilities for nested explanation results, compatible with the
ResultAnalyzer data structure:

results[value_name][characteristic_name][state][feature_index] = value

Supports scalar feature values and vector-valued feature values, e.g. policy
vectors. For vector-valued values, you can select one component/action via
`component`, or aggregate the vector via `vector_mode`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from result_calculator import ResultCalculator, Results, State


class ResultVisualizer:
    """
    Plot heatmaps for explanation results such as Tau and Shapley values.

    Expected structure:
        results[value_name][characteristic_name][state][feature_index] = value

    Example:
        visualizer = ResultVisualizer(
            analyzer.results,
            feature_names=["x", "y"],
            digits=3,
        )

        visualizer.plot_heatmaps(output_dir="plots/raw")
        visualizer.plot_difference_heatmaps(
            left_value="tau",
            right_value="shapley",
            output_dir="plots/diff",
            print_values=True,
        )
    """

    def __init__(self, results: Results, feature_names: Optional[List[str]] = None, digits: int = 3) -> None:
        self.calculator = ResultCalculator(results=results, feature_names=feature_names, digits=digits)
        self.feature_names = feature_names
        self.digits = digits

    @property
    def results(self):
        return self.calculator.results

    # ------------------------------------------------------------------
    # Public plotting methods

    def plot_heatmaps(
        self,
        value_names: Optional[Sequence[str]] = None,
        characteristic_names: Optional[Sequence[str]] = None,
        component: Optional[int] = None,
        vector_mode: str = "mean",
        output_dir: Optional[Union[str, Path]] = "plots/raw",
        show: bool = False,
        print_values: bool = False,
        file_format: str = "png",
        annotate: bool = True,
    ) -> Dict[Tuple[str, str], np.ndarray]:
        """
        Plot one heatmap per value type and characteristic.

        Rows are states, columns are features, cell values are explanation values.

        Args:
            value_names: Which value types to plot, e.g. ["tau", "shapley"].
                If None, all available value types are used.
            characteristic_names: Which characteristics to plot, e.g. ["local"].
                If None, all available characteristics for each value type are used.
            component: For vector-valued entries, choose one component/action.
                Example: component=0 for action 0. Required if vector_mode="component"
                and vector values occur.
            vector_mode: How to reduce vector values:
                - "component": select one component/action.
                - "mean": mean over vector components.
                - "sum": sum over vector components.
                - "norm": L2 norm of vector.
                - "max_abs": component with largest absolute magnitude.
            output_dir: If provided, saves figures there.
            show: Whether to display figures interactively.
            print_values: Whether to print the numeric matrix to terminal.
            file_format: File extension, e.g. "png", "pdf", "svg".
            annotate: Whether to print cell values inside heatmap cells.

        Returns:
            Dict mapping (value_name, characteristic_name) to the plotted matrix.
        """
        matrices: Dict[Tuple[str, str], np.ndarray] = {}

        for value_name in self.calculator.select_value_names(value_names):
            characteristic_dict = self.results[value_name]
            selected_characteristics = self.calculator.select_characteristic_names(
                characteristic_dict, characteristic_names
            )

            for characteristic_name in selected_characteristics:
                values = characteristic_dict[characteristic_name]
                states, matrix = self.calculator.state_values_to_matrix(
                    values, component=component, vector_mode=vector_mode
                )

                matrices[(value_name, characteristic_name)] = matrix

                title = f"{value_name.upper()} - {characteristic_name}"
                if component is not None:
                    title += f" - component/action {component}"
                elif self.calculator.contains_vector_values(values):
                    title += f" - {vector_mode}"

                if print_values:
                    self._print_matrix(title, states, matrix)

                self._plot_matrix(
                    matrix=matrix,
                    states=states,
                    title=title,
                    output_dir=output_dir,
                    filename=self.calculator.safe_filename(title, file_format),
                    show=show,
                    annotate=annotate,
                )

        return matrices

    def plot_difference_heatmaps(
        self,
        left_value: str,
        right_value: str,
        characteristic_names: Optional[Sequence[str]] = None,
        component: Optional[int] = None,
        vector_mode: str = "mean",
        output_dir: Optional[Union[str, Path]] = "plots/diff",
        show: bool = False,
        print_values: bool = False,
        file_format: str = "png",
        annotate: bool = True,
        absolute: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Plot difference heatmaps between two value types.

        Difference is computed as:
            left_value - right_value

        Example:
            plot_difference_heatmaps("tau", "shapley")

        Args:
            left_value: First value type, e.g. "tau".
            right_value: Second value type, e.g. "shapley".
            characteristic_names: Characteristics to compare. If None, compares all
                characteristics available in both value types.
            component: For vector-valued entries, choose one component/action.
            vector_mode: How to reduce vector values if component is not used.
            output_dir: If provided, saves figures there.
            show: Whether to display figures interactively.
            print_values: Whether to print the numeric difference matrix.
            file_format: File extension, e.g. "png", "pdf", "svg".
            annotate: Whether to print cell values inside heatmap cells.
            absolute: If True, plot abs(left - right) instead of signed difference.
        Returns:
            Dict mapping characteristic_name to the difference matrix.
        """
        if left_value not in self.results:
            raise KeyError(f"Unknown value type: {left_value}")
        if right_value not in self.results:
            raise KeyError(f"Unknown value type: {right_value}")

        left_chars = set(self.results[left_value].keys())
        right_chars = set(self.results[right_value].keys())
        common_chars = sorted(left_chars & right_chars)

        if characteristic_names is not None:
            common_chars = [c for c in characteristic_names if c in common_chars]

        if not common_chars:
            raise ValueError(
                f"No common characteristics found between {left_value} and {right_value}."
            )

        differences: Dict[str, np.ndarray] = {}
        common_chars = self.calculator.common_characteristics(left_value, right_value, characteristic_names)

        for characteristic_name in common_chars:
            states, diff_matrix = self.calculator.difference_matrix(
                left_value=left_value,
                right_value=right_value,
                characteristic_name=characteristic_name,
                component=component,
                vector_mode=vector_mode,
                absolute=absolute,
            )
            differences[characteristic_name] = diff_matrix

            left_values = self.results[left_value][characteristic_name]
            right_values = self.results[right_value][characteristic_name]

            title = f"DIFF {left_value.upper()} - {right_value.upper()} - {characteristic_name}"
            if absolute:
                title = f"ABS {title}"
            if component is not None:
                title += f" - component/action {component}"
            elif self.calculator.contains_vector_values(left_values) or self.calculator.contains_vector_values(right_values):
                title += f" - {vector_mode}"

            if print_values:
                self._print_matrix(title, states, diff_matrix)

            self._plot_matrix(
                matrix=diff_matrix,
                states=states,
                title=title,
                output_dir=output_dir,
                filename=self.calculator.safe_filename(title, file_format),
                show=show,
                annotate=annotate,
                centered=True,
            )

        return differences

    def plot_value_comparison_bars(
        self,
        characteristic_names=None,
        states=None,
        features=None,
        output_dir="plots/value_comparison",
        show=False,
        file_format="png",
        print_values=False,
    ):
        """
        Plot value-method comparison bars for fixed characteristic, state and feature.

        For each characteristic/state/feature combination, one plot is created.
        Bars show the individual value types, e.g. Tau and Shapley.
        Horizontal lines show mean, min and max across value types.
        The shaded band shows mean ± std.

        Args:
            characteristic_names: Characteristics to plot. If None, uses all available.
            states: States to plot. If None, uses all available states.
            features: Feature indices to plot. If None, uses all features.
            output_dir: If provided, saves figures there.
            show: Whether to display figures interactively.
            file_format: File extension, e.g. "png", "pdf", "svg".
            print_values: Whether to print numeric values.

        Returns:
            Dict mapping (characteristic, state, feature) to plotted statistics.
        """
        output = {}
        all_value_names = list(self.results.keys())
        all_characteristics = sorted(set().union(*(self.results[v].keys() for v in all_value_names)))
        selected_characteristics = all_characteristics if characteristic_names is None else list(characteristic_names)

        for characteristic_name in selected_characteristics:
            value_names = [v for v in all_value_names if characteristic_name in self.results[v]]
            if not value_names:
                continue

            all_states = sorted(
                set().union(*[set(self.results[v][characteristic_name].keys()) for v in value_names]),
                key=str,
            )
            selected_states = all_states if states is None else [self.calculator.normalize_state(s) for s in states]

            for state in selected_states:
                available_value_names = [v for v in value_names if state in self.results[v][characteristic_name]]
                if not available_value_names:
                    continue

                num_features = len(self.results[available_value_names[0]][characteristic_name][state])
                selected_features = range(num_features) if features is None else features

                for feature_idx in selected_features:
                    labels, raw_values = [], []

                    for value_name in available_value_names:
                        value = self.results[value_name][characteristic_name][state][feature_idx]
                        scalar = self.calculator.scalarize_value(value, component=None, vector_mode="mean")
                        labels.append(value_name)
                        raw_values.append(scalar)

                    values_arr = self.calculator.clean_small_values(np.asarray(raw_values, dtype=float))
                    mean = float(np.mean(values_arr))
                    std = float(np.std(values_arr))
                    min_value = float(np.min(values_arr))
                    max_value = float(np.max(values_arr))

                    stats = {
                        "values": dict(zip(labels, values_arr.tolist())),
                        "mean": mean,
                        "std": std,
                        "min": min_value,
                        "max": max_value,
                    }
                    output[(characteristic_name, state, feature_idx)] = stats

                    title = (
                        f"VALUE COMPARISON - {characteristic_name} - "
                        f"State {self.calculator.format_state(state)} - {self.calculator.feature_name(feature_idx)}"
                    )

                    if print_values:
                        print("\n" + "=" * 100)
                        print(title)
                        print("=" * 100)
                        for label, value in zip(labels, values_arr):
                            print(f"{label:<16} {value:>14.{self.digits}f}")
                        print("-" * 100)
                        print(f"{'mean':<16} {mean:>14.{self.digits}f}")
                        print(f"{'std':<16} {std:>14.{self.digits}f}")
                        print(f"{'min':<16} {min_value:>14.{self.digits}f}")
                        print(f"{'max':<16} {max_value:>14.{self.digits}f}")

                    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2 + 3), 5))
                    x = np.arange(len(labels))
                    ax.bar(x, values_arr)

                    ax.axhline(mean, linestyle="--", linewidth=1.2, label=f"mean = {mean:.{self.digits}f}")
                    ax.axhline(min_value, linestyle=":", linewidth=1.0, label=f"min = {min_value:.{self.digits}f}")
                    ax.axhline(max_value, linestyle=":", linewidth=1.0, label=f"max = {max_value:.{self.digits}f}")
                    ax.axhspan(mean - std, mean + std, alpha=0.15, label=f"mean ± std = {mean:.{self.digits}f} ± {std:.{self.digits}f}")

                    ax.set_title(title)
                    ax.set_xlabel("Value type")
                    ax.set_ylabel("Value")
                    ax.set_xticks(x)
                    ax.set_xticklabels(labels)
                    ax.legend()

                    for idx, value in enumerate(values_arr):
                        ax.text(idx, value, f"{value:.{self.digits}f}", ha="center", va="bottom" if value >= 0 else "top")

                    fig.tight_layout()

                    if output_dir is not None:
                        output_path = Path(output_dir)
                        output_path.mkdir(parents=True, exist_ok=True)
                        filename = self.calculator.safe_filename(
                            f"value_comparison_{characteristic_name}_state_{self.calculator.format_state(state, compact=True)}_{self.calculator.feature_name(feature_idx)}",
                            file_format,
                        )
                        fig.savefig(output_path / filename, bbox_inches="tight")

                    if show:
                        plt.show()
                    else:
                        plt.close(fig)

        return output

    def plot_action_bars(
        self,
        value_name: str,
        characteristic_name: str,
        states: Optional[Sequence[State]] = None,
        output_dir: Optional[Union[str, Path]] = "plots/action_bars",
        show: bool = False,
        file_format: str = "png",
        action_names: Optional[Sequence[str]] = None,
        print_values: bool = True,
    ) -> Dict[State, np.ndarray]:
        """Plot grouped bar charts for vector-valued feature contributions."""
        if value_name not in self.results:
            raise KeyError(f"Unknown value type: {value_name}")
        if characteristic_name not in self.results[value_name]:
            raise KeyError(f"Unknown characteristic '{characteristic_name}' for value type '{value_name}'")

        values = self.results[value_name][characteristic_name]
        available_states = sorted([self.calculator.normalize_state(s) for s in values.keys()], key=str)
        selected_states = available_states if states is None else [self.calculator.normalize_state(s) for s in states]
        matrices: Dict[State, np.ndarray] = {}

        for state in selected_states:
            if state not in values:
                raise KeyError(f"State {state} not found in {value_name}/{characteristic_name}")

            matrix = []
            for value in values[state]:
                arr = np.asarray(value, dtype=float)
                if arr.ndim == 0:
                    raise ValueError("plot_action_bars requires vector-valued feature values.")
                matrix.append(arr.flatten())

            matrix = self.calculator.clean_small_values(np.asarray(matrix, dtype=float))
            matrices[state] = matrix
            num_features, num_actions = matrix.shape

            labels = [f"Action {i}" for i in range(num_actions)] if action_names is None else list(action_names)
            if len(labels) != num_actions:
                raise ValueError(f"action_names has length {len(labels)}, but vector has {num_actions} components.")

            if print_values:
                print("\n" + "=" * 100)
                print(f"ACTION BARS - {value_name.upper()} - {characteristic_name} - State {self.calculator.format_state(state)}")
                print("=" * 100)
                header = f"{'Feature':<16}" + "".join(f"{label:>14}" for label in labels)
                print(header)
                print("-" * 100)
                for feature_idx in range(num_features):
                    row = f"{self.calculator.feature_name(feature_idx):<16}" + "".join(
                        f"{matrix[feature_idx, action_idx]:>14.{self.digits}f}"
                        for action_idx in range(num_actions)
                    )
                    print(row)

            x = np.arange(num_actions)
            width = 0.8 / max(num_features, 1)
            fig, ax = plt.subplots(figsize=(max(7, 1.2 * num_actions + 3), 5))

            for feature_idx in range(num_features):
                offset = (feature_idx - (num_features - 1) / 2) * width
                ax.bar(x + offset, matrix[feature_idx], width, label=self.calculator.feature_name(feature_idx))

            ax.axhline(0.0, linewidth=0.8)
            ax.set_title(f"{value_name.upper()} - {characteristic_name} - State {self.calculator.format_state(state)}")
            ax.set_xlabel("Action / Component")
            ax.set_ylabel("Feature contribution")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend(title="Feature")
            fig.tight_layout()

            if output_dir is not None:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                filename = self.calculator.safe_filename(
                    f"action_bars_{value_name}_{characteristic_name}_state_{self.calculator.format_state(state, compact=True)}",
                    file_format,
                )
                fig.savefig(output_path / filename, bbox_inches="tight")

            if show:
                plt.show()
            else:
                plt.close(fig)

        return matrices

    # ------------------------------------------------------------------
    # Plotting helpers

    def _plot_matrix(
        self,
        matrix: np.ndarray,
        states: List[State],
        title: str,
        output_dir: Optional[Union[str, Path]],
        filename: str,
        show: bool,
        annotate: bool,
        centered: bool = False,
    ) -> None:
        if matrix.size == 0:
            return

        fig_width = max(6, 1.2 * matrix.shape[1] + 3)
        fig_height = max(4, 0.45 * matrix.shape[0] + 2)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        if centered:
            max_abs = np.nanmax(np.abs(matrix))
            vmin, vmax = -max_abs, max_abs
        else:
            vmin, vmax = None, None

        im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax)

        ax.set_title(title)
        ax.set_xlabel("Feature")
        ax.set_ylabel("State")
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels([self.calculator.feature_name(i) for i in range(matrix.shape[1])])
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels([self.calculator.format_state(s) for s in states])

        if annotate:
            for row_idx in range(matrix.shape[0]):
                for col_idx in range(matrix.shape[1]):
                    ax.text(col_idx, row_idx, f"{matrix[row_idx, col_idx]:.{self.digits}f}", ha="center", va="center")

        fig.tight_layout()

        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path / filename, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def _print_matrix(self, title: str, states: List[State], matrix: np.ndarray) -> None:
        print("\n" + "=" * 100)
        print(title)
        print("=" * 100)
        header = f"{'State':<18}" + "".join(f"{self.calculator.feature_name(i):>14}" for i in range(matrix.shape[1]))
        print(header)
        print("-" * 100)

        for state, row in zip(states, matrix):
            row_text = f"{self.calculator.format_state(state):<18}" + "".join(
                f"{float(value):>14.{self.digits}f}" for value in row
            )
            print(row_text)
