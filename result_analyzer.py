"""
result_analyzer.py

Utility class for analyzing and exporting explanation results such as
Shapley values, Tau values, SVERL characteristics, policy-vector values, etc.

Expected input format:

results = {
    "tau": {
        "local": {
            (0, 0): [0.0, 0.1],
            (0, 1): [0.2, 0.3],
        },
        "global": {
            (0, 0): [0.0, 0.1],
        },
    },
    "shapley": {
        "local": {
            (0, 0): [0.0, 0.1],
        }
    }
}

Each state maps to a list of feature values.
Each feature value may be either:
    - scalar, e.g. 0.25
    - numpy scalar
    - vector/list/array, e.g. policy-action contributions [0.1, -0.2, 0.0, 0.1]
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from result_calculator import ResultCalculator, Results, State, StateValues


class ResultAnalyzer:
    """
    Analyze, print, summarize and export nested explanation results.

    Expected structure:
        results[value_name][characteristic_name][state][feature_index] = value

    Example:
        analyzer = ResultAnalyzer(result, feature_names=["x", "y"])
        analyzer.print_all()
        analyzer.save_pickle("results.pkl")
        analyzer.save_json("results.json")
        analyzer.save_csv("results.csv")
        summary = analyzer.summary()
        analyzer.print_summary()
    """

    def __init__(self, results: Optional[Results] = None, feature_names: Optional[List[str]] = None, digits: int = 3) -> None:
        self.calculator = ResultCalculator(results=results, feature_names=feature_names, digits=digits)
        self.feature_names = feature_names
        self.digits = digits

    @property
    def results(self) -> Dict[str, Dict[str, Dict[State, List[Any]]]]:
        return self.calculator.results

    # ------------------------------------------------------------------
    # Data management

    def add_result(self, value_name: str, characteristic_name: str, values: StateValues) -> None:
        """
        Add one result table.

        Args:
            value_name: e.g. "tau" or "shapley".
            characteristic_name: e.g. "local", "global", "policy", "value_function".
            values: dict mapping state -> list of feature values.
        """
        self.calculator.add_result(value_name, characteristic_name, values)

    def add_results(self, results: Results) -> None:
        """Add many nested results at once."""
        self.calculator.add_results(results)

    # ------------------------------------------------------------------
    # Printing

    def print_all(self) -> None:
        """Print all available value types and characteristics to the terminal."""
        for value_name, characteristic_dict in self.results.items():
            for characteristic_name, values in characteristic_dict.items():
                title = f"{value_name.upper()} Values - {characteristic_name}"
                self.print_feature_table(title, values)

    def print_feature_table(self, title: str, values: StateValues) -> None:
        """Print one state -> feature table to the terminal."""
        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

        for state, feature_values in values.items():
            print(f"\nState {self.calculator.format_state(state)}")
            print("-" * 80)
            print(f"{'Feature':<16} {'Value':>30}")
            print("-" * 80)

            for i, value in enumerate(feature_values):
                name = self.calculator.feature_name(i)
                print(f"{name:<16} {self.calculator.format_value(value):>30}")

    def print_per_feature_summary(self) -> None:
        """Print per-feature summary statistics for all results across states."""
        summary = self.per_feature_summary()

        for value_name, characteristic_dict in summary.items():
            for characteristic_name, feature_stats in characteristic_dict.items():
                title = f"SUMMARY - {value_name.upper()} - {characteristic_name}"
                print("\n" + "=" * 80)
                print(title)
                print("=" * 80)
                print(f"{'Feature':<16} {'Mean':>18} {'Std':>18} {'Min':>18} {'Max':>18} {'Count':>10}")
                print("-" * 80)

                for feature_idx, stats in feature_stats.items():
                    print(
                        f"{self.calculator.feature_name(feature_idx):<16} "
                        f"{self.calculator.format_value(stats['mean']):>18} "
                        f"{self.calculator.format_value(stats['std']):>18} "
                        f"{self.calculator.format_value(stats['min']):>18} "
                        f"{self.calculator.format_value(stats['max']):>18} "
                        f"{stats['count']:>10}"
                    )

    def print_per_state_summary(self) -> None:
        """Print per-state summary statistics across features for all results."""
        summary = self.per_state_summary()

        for value_name, characteristic_dict in summary.items():
            for characteristic_name, state_dict in characteristic_dict.items():
                title = f"PER STATE SUMMARY - {value_name.upper()} - {characteristic_name}"
                print("\n" + "=" * 100)
                print(title)
                print("=" * 100)
                print(
                    f"{'State':<18} {'MeanFeat':>18} {'StdFeat':>18} "
                    f"{'SumFeat':>18} {'MinFeat':>18} {'MaxFeat':>18} {'#Feat':>8}"
                )
                print("-" * 100)

                for state, stats in state_dict.items():
                    print(
                        f"{self.calculator.format_state(state):<18} "
                        f"{self.calculator.format_value(stats['mean_across_features']):>18} "
                        f"{self.calculator.format_value(stats['std_across_features']):>18} "
                        f"{self.calculator.format_value(stats['sum_across_features']):>18} "
                        f"{self.calculator.format_value(stats['min_across_features']):>18} "
                        f"{self.calculator.format_value(stats['max_across_features']):>18} "
                        f"{stats['num_features']:>8}"
                    )

    def print_summary_across_values(self) -> None:
        """Print summary statistics per state, characteristic and feature across all value types."""
        summary = self.summary_across_values()

        for characteristic_name, state_dict in summary.items():
            for state, feature_stats in state_dict.items():
                title = f"SUMMARY ACROSS VALUES - {characteristic_name} - State {self.calculator.format_state(state)}"
                print("\n" + "=" * 80)
                print(title)
                print("=" * 80)
                print(f"{'Feature':<16} {'Mean':>18} {'Std':>18} {'Min':>18} {'Max':>18} {'Count':>10}")
                print("-" * 80)

                for feature_idx, stats in feature_stats.items():
                    print(
                        f"{self.calculator.feature_name(feature_idx):<16} "
                        f"{self.calculator.format_value(stats['mean']):>18} "
                        f"{self.calculator.format_value(stats['std']):>18} "
                        f"{self.calculator.format_value(stats['min']):>18} "
                        f"{self.calculator.format_value(stats['max']):>18} "
                        f"{stats['count']:>10}"
                    )

    def print_value_comparison(
        self,
        left_value:str,
        right_value: str,
        characteristic_names: Optional[List[str]] = None,
        absolute: bool = False,
    ) -> None:
        """Print comparison left_value - right_value."""
        comparison = self.compare_values(left_value, right_value, characteristic_names, absolute)
        label = f"ABS({left_value} - {right_value})" if absolute else f"{left_value} - {right_value}"

        for characteristic_name, state_dict in comparison.items():
            title = f"VALUE COMPARISON - {label} - {characteristic_name}"
            self.print_feature_table(title, state_dict)

    # ------------------------------------------------------------------
    # Calculations delegated to ResultCalculator

    def per_feature_summary(self) -> Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]:
        """
        Compute mean, std, min, max and count per value type,
        characteristic and feature across all states.

        For scalar feature values, the statistics are scalars.
        For vector feature values, the statistics are computed component-wise.
        """
        return self.calculator.per_feature_summary()

    def per_state_summary(self) -> Dict[str, Dict[str, Dict[State, Dict[str, Any]]]]:
        """
        Compute summary statistics across features for each state.

        Useful if you want to know how much total explanation mass or variation
        a state has.
        """
        return self.calculator.per_state_summary()

    def summary_across_values(self) -> Dict[str, Dict[State, Dict[int, Dict[str, Any]]]]:
        """
        Compute mean, std, min, max and count per characteristic, state and feature
        across all value types.

        Example:
            If results contain both "tau" and "shapley", this computes statistics
            over those value types for the same characteristic/state/feature.

        Output structure:
            output[characteristic_name][state][feature_idx] = {
                "mean": ...,
                "std": ...,
                "min": ...,
                "max": ...,
                "count": ...
            }
        """
        return self.calculator.summary_across_values()

    def compare_values(
        self,
        left_value: str,
        right_value: str,
        characteristic_names: Optional[List[str]] = None,
        absolute: bool = False,
    ) -> Dict[str, Dict[State, List[Any]]]:
        """
        Compare two value types by computing left_value - right_value.

        Args:
            left_value: First value type, e.g. "tau".
            right_value: Second value type, e.g. "shapley".
            characteristic_names: Characteristics to compare. If None, compares all common characteristics.
            absolute: If True, computes abs(left_value - right_value).

        Returns:
            Dict[characteristic][state][feature_index] = difference.
        """
        return self.calculator.compare_values(left_value, right_value, characteristic_names, absolute)

    # ------------------------------------------------------------------
    # Export

    def _ensure_parent(self, output_dir: Union[str, Path]) -> None:
        Path(output_dir).parent.mkdir(parents=True, exist_ok=True)

    def save_pickle(self, output_dir: Union[str, Path] = "data/results.pkl") -> None:
        """Save raw results as pickle."""
        self._ensure_parent(output_dir)
        with open(output_dir, "wb") as file:
            pickle.dump(self.calculator.pythonify(self.results), file)

    def save_per_feature_summary_pickle(self, output_dir: Union[str, Path] = "data/per_feature_summary.pkl") -> None:
        """Save computed per-feature summary as pickle."""

        self._ensure_parent(output_dir)

        with open(output_dir, "wb") as file:
            pickle.dump(self.calculator.pythonify(self.per_feature_summary()), file)

    def save_per_state_summary_pickle(self, output_dir: Union[str, Path] = "data/per_state_summary.pkl") -> None:
        """Save computed per-state summary as pickle."""
        self._ensure_parent(output_dir)
        with open(output_dir, "wb") as file:
            pickle.dump(self.calculator.pythonify(self.per_state_summary()), file)

    def save_across_values_summary_pickle(self, output_dir: Union[str, Path] = "data/summary_across_values.pkl") -> None:
        """Save computed across-values summary as pickle."""
        self._ensure_parent(output_dir)
        with open(output_dir, "wb") as file:
            pickle.dump(self.calculator.pythonify(self.summary_across_values()), file)

    def save_value_comparison_pickle(
        self,
        left_value: str,
        right_value: str,
        output_dir: Union[str, Path] = "data/value_comparison.pkl",
        characteristic_names: Optional[List[str]] = None,
        absolute: bool = False,
    ) -> None:
        """Save comparison left_value -right_value as pickle."""
        comparison = self.compare_values(left_value, right_value, characteristic_names, absolute)
        self._ensure_parent(output_dir)
        with open(output_dir, "wb") as file:
            pickle.dump(self.calculator.pythonify(comparison), file)

    def save_json(self, output_dir: Union[str, Path] = "data/results.json", include_summary: bool = True) -> None:
        """Save rseults, optionally including summary, as JSON."""
        payload: Dict[str, Any] = {"results": self.calculator.jsonify(self.results)}

        if include_summary:
            payload["per_feature_summary"] = self.calculator.jsonify(self.per_feature_summary())
            payload["per_state_summary"] = self.calculator.jsonify(self.per_state_summary())
            payload["summary_across_values"] = self.calculator.jsonify(self.summary_across_values())

        self._ensure_parent(output_dir)
        with open(output_dir, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    def save_csv(self, output_dir: Union[str, Path] = "data/results.csv") -> None:
        """Save flattened result rows to CSV."""
        rows = self.flatten_results()
        self._ensure_parent(output_dir)
        with open(output_dir, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["value_name", "characteristic", "state", "feature_index", "feature_name", "value"],
            )
            writer.writeheader()
            writer.writerows(rows)

    def save_summary_csv(self, output_dir: Union[str, Path] = "data/summary.csv") -> None:
        """
        Save all available summary statistics to one CSV:
            - per_feature_summary()
            - per_state_summary()
            - summary_across_values()
        """
        rows = []

        for value_name, characteristic_dict in self.per_feature_summary().items():
            for characteristic_name, feature_stats in characteristic_dict.items():
                for feature_idx, stats in feature_stats.items():
                    rows.append({
                        "summary_type": "per_feature_summary",
                        "value_name": value_name,
                        "characteristic": characteristic_name,
                        "state": "",
                        "feature_index": feature_idx,
                        "feature_name": self.calculator.feature_name(feature_idx),
                        "mean": self.calculator.to_serializable(stats["mean"]),
                        "std": self.calculator.to_serializable(stats["std"]),
                        "sum": "",
                        "min": self.calculator.to_serializable(stats["min"]),
                        "max": self.calculator.to_serializable(stats["max"]),
                        "count": stats["count"],
                    })

        for value_name, characteristic_dict in self.per_state_summary().items():
            for characteristic_name, state_dict in characteristic_dict.items():
                for state, stats in state_dict.items():
                    rows.append({
                        "summary_type": "per_state_summary",
                        "value_name": value_name,
                        "characteristic": characteristic_name,
                        "state": self.calculator.format_state(state),
                        "feature_index": "",
                        "feature_name": "",
                        "mean": self.calculator.to_serializable(stats["mean_across_features"]),
                        "std": self.calculator.to_serializable(stats["std_across_features"]),
                        "sum": self.calculator.to_serializable(stats["sum_across_features"]),
                        "min": self.calculator.to_serializable(stats["min_across_features"]),
                        "max": self.calculator.to_serializable(stats["max_across_features"]),
                        "count": stats["num_features"],
                    })

        for characteristic_name, state_dict in self.summary_across_values().items():
            for state, feature_dict in state_dict.items():
                for feature_idx, stats in feature_dict.items():
                    rows.append({
                        "summary_type": "summary_across_values",
                        "value_name": "ALL_VALUES",
                        "characteristic": characteristic_name,
                        "state": self.calculator.format_state(state),
                        "feature_index": feature_idx,
                        "feature_name": self.calculator.feature_name(feature_idx),
                        "mean": self.calculator.to_serializable(stats["mean"]),
                        "std": self.calculator.to_serializable(stats["std"]),
                        "sum": "",
                        "min": self.calculator.to_serializable(stats["min"]),
                        "max": self.calculator.to_serializable(stats["max"]),
                        "count": stats["count"],
                    })

        self._ensure_parent(output_dir)
        with open(output_dir, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "summary_type", "value_name", "characteristic", "state",
                    "feature_index", "feature_name", "mean", "std", "sum", "min", "max", "count",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def save_value_comparison_csv(
        self,
        left_value: str,
        right_value: str,
        output_dir: Union[str, Path] = "data/value_comparison.csv",
        characteristic_names: Optional[List[str]] = None,
        absolute: bool = False,
    ) -> None:
        """Save comparison left_value - right_value as CSV."""
        comparison = self.compare_values(left_value, right_value, characteristic_names, absolute)
        rows = []

        for characteristic_name, state_dict in comparison.items():
            for state, feature_values in state_dict.items():
                for feature_idx, value in enumerate(feature_values):
                    rows.append({
                        "comparison": f"abs({left_value}-{right_value})" if absolute else f"{left_value}-{right_value}",
                        "characteristic": characteristic_name,
                        "state": self.calculator.format_state(state),
                        "feature_index": feature_idx,
                        "feature_name": self.calculator.feature_name(feature_idx),
                        "value": self.calculator.to_serializable(value),
                    })

        self._ensure_parent(output_dir)
        with open(output_dir, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["comparison", "characteristic", "state", "feature_index", "feature_name", "value"],
            )
            writer.writeheader()
            writer.writerows(rows)

    # ------------------------------------------------------------------
    # Flattening

    def flatten_results(self) -> List[Dict[str, Any]]:
        """Flatten allresults into row dictionaries."""
        rows = []

        for value_name, characteristic_dict in self.results.items():
            for characteristic_name, values in characteristic_dict.items():
                for state, feature_values in values.items():
                    for feature_idx, value in enumerate(feature_values):
                        rows.append({
                            "value_name": value_name,
                            "characteristic": characteristic_name,
                            "state": self.calculator.format_state(state),
                            "feature_index": feature_idx,
                            "feature_name": self.calculator.feature_name(feature_idx),
                            "value": self.calculator.to_serializable(value),
                        })

        return rows