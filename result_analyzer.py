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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np


State = Tuple[Any, ...]
FeatureValue = Union[float, int, np.ndarray, List[float], Tuple[float, ...]]
StateValues = Mapping[State, List[FeatureValue]]
Results = Mapping[str, Mapping[str, StateValues]]


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
        self.results: Dict[str, Dict[str, Dict[State, List[Any]]]] = {}
        self.feature_names = feature_names
        self.digits = digits

        if results is not None:
            self.add_results(results)

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
        # Normalize state keys to tuples and ensure feature values are lists.
        self.results.setdefault(value_name, {})[characteristic_name] = {
            self._normalize_state(state): list(feature_values)
            for state, feature_values in values.items()
        }


    def add_results(self, results: Results) -> None:
        """Add many nested results at once."""
        for value_name, characteristic_dict in results.items():
            for characteristic_name, values in characteristic_dict.items():
                self.add_result(value_name, characteristic_name, values)

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
            print(f"\nState {state}")
            print("-" * 80)
            print(f"{'Feature':<16} {'Value':>30}")
            print("-" * 80)

            for i, value in enumerate(feature_values):
                name = self._feature_name(i)
                print(f"{name:<16} {self.format_value(value):>30}")

    def print_summary(self) -> None:
        """Print summary statistics for all results."""
        summary = self.summary()

        for value_name, characteristic_dict in summary.items():
            for characteristic_name, feature_stats in characteristic_dict.items():
                title = f"SUMMARY - {value_name.upper()} - {characteristic_name}"
                print("\n" + "=" * 80)
                print(title)
                print("=" * 80)
                print(
                    f"{'Feature':<16} {'Mean':>18} {'Std':>18} {'Min':>18} {'Max':>18} {'Count':>10}"
                )
                print("-" * 80)

                for feature_idx, stats in feature_stats.items():
                    name = self._feature_name(feature_idx)
                    print(
                        f"{name:<16} "
                        f"{self.format_value(stats['mean']):>18} "
                        f"{self.format_value(stats['std']):>18} "
                        f"{self.format_value(stats['min']):>18} "
                        f"{self.format_value(stats['max']):>18} "
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
                    f"{'State':<18} "
                    f"{'MeanFeat':>18} "
                    f"{'StdFeat':>18} "
                    f"{'SumFeat':>18} "
                    f"{'MinFeat':>18} "
                    f"{'MaxFeat':>18} "
                    f"{'#Feat':>8}"
                )
                print("-" * 100)

                for state, stats in state_dict.items():
                    print(
                        f"{str(state):<18} "
                        f"{self.format_value(stats['mean_across_features']):>18} "
                        f"{self.format_value(stats['std_across_features']):>18} "
                        f"{self.format_value(stats['sum_across_features']):>18} "
                        f"{self.format_value(stats['min_across_features']):>18} "
                        f"{self.format_value(stats['max_across_features']):>18} "
                        f"{stats['num_features']:>8}"
                    )

    def print_summary_across_values(self) -> None:
        """Print summary statistics per state, characteristic and feature across all value types."""
        summary = self.summary_across_values()

        for characteristic_name, state_dict in summary.items():
            for state, feature_stats in state_dict.items():
                title = f"SUMMARY ACROSS VALUES - {characteristic_name} - State {state}"

                print("\n" + "=" * 80)
                print(title)
                print("=" * 80)
                print(
                    f"{'Feature':<16} {'Mean':>18} {'Std':>18} "
                    f"{'Min':>18} {'Max':>18} {'Count':>10}"
                )
                print("-" * 80)

                for feature_idx, stats in feature_stats.items():
                    name = self._feature_name(feature_idx)

                    print(
                        f"{name:<16} "
                        f"{self.format_value(stats['mean']):>18} "
                        f"{self.format_value(stats['std']):>18} "
                        f"{self.format_value(stats['min']):>18} "
                        f"{self.format_value(stats['max']):>18} "
                        f"{stats['count']:>10}"
                    )

    # ------------------------------------------------------------------
    # Statistics

    def summary(self) -> Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]:
        """
        Compute mean, std, min, max and count per value type,
        characteristic and feature across all states.

        For scalar feature values, the statistics are scalars.
        For vector feature values, the statistics are computed component-wise.
        """
        output: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}

        for value_name, characteristic_dict in self.results.items():
            output[value_name] = {}

            for characteristic_name, values in characteristic_dict.items():
                grouped = self._group_by_feature(values)
                output[value_name][characteristic_name] = {}

                for feature_idx, feature_values in grouped.items():
                    arr = self._stack_values(feature_values)
                    output[value_name][characteristic_name][feature_idx] = {
                        "mean": np.mean(arr, axis=0),
                        "std": np.std(arr, axis=0),
                        "min": np.min(arr, axis=0),
                        "max": np.max(arr, axis=0),
                        "count": int(arr.shape[0]),
                    }

        return output

    def per_state_summary(self) -> Dict[str, Dict[str, Dict[State, Dict[str, Any]]]]:
        """
        Compute summary statistics across features for each state.

        Useful if you want to know how much total explanation mass or variation
        a state has.
        """
        output: Dict[str, Dict[str, Dict[State, Dict[str, Any]]]] = {}

        for value_name, characteristic_dict in self.results.items():
            output[value_name] = {}

            for characteristic_name, values in characteristic_dict.items():
                output[value_name][characteristic_name] = {}

                for state, feature_values in values.items():
                    arr = self._stack_values(feature_values)
                    output[value_name][characteristic_name][state] = {
                        "mean_across_features": np.mean(arr, axis=0),
                        "std_across_features": np.std(arr, axis=0),
                        "sum_across_features": np.sum(arr, axis=0),
                        "min_across_features": np.min(arr, axis=0),
                        "max_across_features": np.max(arr, axis=0),
                        "num_features": int(arr.shape[0]),
                    }

        return output
    

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
        grouped: Dict[str, Dict[State, Dict[int, List[Any]]]] = {}

        for _value_name, characteristic_dict in self.results.items():
            for characteristic_name, values in characteristic_dict.items():
                grouped.setdefault(characteristic_name, {})

                for state, feature_values in values.items():
                    state = self._normalize_state(state)
                    grouped[characteristic_name].setdefault(state, {})

                    for feature_idx, value in enumerate(feature_values):
                        grouped[characteristic_name][state].setdefault(feature_idx, [])
                        grouped[characteristic_name][state][feature_idx].append(value)

        output: Dict[str, Dict[State, Dict[int, Dict[str, Any]]]] = {}

        for characteristic_name, state_dict in grouped.items():
            output[characteristic_name] = {}

            for state, feature_dict in state_dict.items():
                output[characteristic_name][state] = {}

                for feature_idx, values in feature_dict.items():
                    arr = self._stack_values(values)

                    output[characteristic_name][state][feature_idx] = {
                        "mean": np.mean(arr, axis=0),
                        "std": np.std(arr, axis=0),
                        "min": np.min(arr, axis=0),
                        "max": np.max(arr, axis=0),
                        "count": int(arr.shape[0]),
                    }

        return output

    # ------------------------------------------------------------------
    # Export

    def save_pickle(self, output_dir: Union[str, Path] = "data/results.pkl") -> None:
        """Save raw results as pickle."""
        if output_dir is not None:
            output_path = Path(output_dir).parent
            output_path.mkdir(parents=True, exist_ok=True)
        with open(output_dir, "wb") as file:
            pickle.dump(self.results, file)

    def save_summary_pickle(self, output_dir: Union[str, Path] = "data/summary.pkl") -> None:
        """Save computed summary as pickle."""
        if output_dir is not None:
            output_path = Path(output_dir).parent
            output_path.mkdir(parents=True, exist_ok=True)
        with open(output_dir, "wb") as file:
            pickle.dump(self.summary(), file)

    def save_per_state_summary_pickle(self, output_dir: Union[str, Path] = "data/per_state_summary.pkl") -> None:
        """Save computed per-state summary as pickle."""
        if output_dir is not None:
            output_path = Path(output_dir).parent
            output_path.mkdir(parents=True, exist_ok=True)
        with open(output_dir, "wb") as file:
            pickle.dump(self.per_state_summary(), file)

    def save_across_values_summary_pickle(self, output_dir: Union[str, Path] = "data/summary_across_values.pkl") -> None:
        """Save computed across-values summary as pickle."""
        if output_dir is not None:
            output_path = Path(output_dir).parent
            output_path.mkdir(parents=True, exist_ok=True)
        with open(output_dir, "wb") as file:
            pickle.dump(self.summary_across_values(), file)

    def save_json(self, output_dir: Union[str, Path] = "data/results.json", include_summary: bool = True) -> None:
        """Save results, optionally including summary, as JSON."""
        payload: Dict[str, Any] = {"results": self._jsonify(self.results)}

        if include_summary:
            payload["summary"] = self._jsonify(self.summary())
            payload["per_state_summary"] = self._jsonify(self.per_state_summary())
            payload["summary_across_values"] = self._jsonify(self.summary_across_values())

        if output_dir is not None:
            output_path = Path(output_dir).parent
            output_path.mkdir(parents=True, exist_ok=True)

        with open(output_dir, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, ensure_ascii=False)

    def save_csv(self, output_dir: Union[str, Path] = "data/results.csv") -> None:
        """
        Save flattened result rows to CSV.

        Vector values are written as JSON-like lists in the 'value' column.
        """
        rows = self.flatten_results()

        if output_dir is not None:
            output_path = Path(output_dir).parent
            output_path.mkdir(parents=True, exist_ok=True)

        with open(output_dir, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "value_name",
                    "characteristic",
                    "state",
                    "feature_index",
                    "feature_name",
                    "value",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    def save_summary_csv(self, output_dir: Union[str, Path] = "data/summary.csv") -> None:
        """
        Save all available summary statistics to one CSV:
            - summary()
            - per_state_summary()
            - summary_across_values()
        """
        rows = []

        # --------------------------------------------------
        # 1. summary()
        normal_summary = self.summary()

        for value_name, characteristic_dict in normal_summary.items():
            for characteristic_name, feature_stats in characteristic_dict.items():
                for feature_idx, stats in feature_stats.items():
                    rows.append(
                        {
                            "summary_type": "summary",
                            "value_name": value_name,
                            "characteristic": characteristic_name,
                            "state": "",
                            "feature_index": feature_idx,
                            "feature_name": self._feature_name(feature_idx),
                            "mean": self._to_serializable(stats["mean"]),
                            "std": self._to_serializable(stats["std"]),
                            "sum": "",
                            "min": self._to_serializable(stats["min"]),
                            "max": self._to_serializable(stats["max"]),
                            "count": stats["count"],
                        }
                    )

        # --------------------------------------------------
        # 2. per_state_summary()
        per_state = self.per_state_summary()

        for value_name, characteristic_dict in per_state.items():
            for characteristic_name, state_dict in characteristic_dict.items():
                for state, stats in state_dict.items():
                    rows.append(
                        {
                            "summary_type": "per_state_summary",
                            "value_name": value_name,
                            "characteristic": characteristic_name,
                            "state": str(state),
                            "feature_index": "",
                            "feature_name": "",
                            "mean": self._to_serializable(stats["mean_across_features"]),
                            "std": self._to_serializable(stats["std_across_features"]),
                            "sum": self._to_serializable(stats["sum_across_features"]),
                            "min": self._to_serializable(stats["min_across_features"]),
                            "max": self._to_serializable(stats["max_across_features"]),
                            "count": stats["num_features"],
                        }
                    )

        # --------------------------------------------------
        # 3. summary_across_values()
        across_values = self.summary_across_values()

        for characteristic_name, state_dict in across_values.items():
            for state, feature_dict in state_dict.items():
                for feature_idx, stats in feature_dict.items():
                    rows.append(
                        {
                            "summary_type": "summary_across_values",
                            "value_name": "ALL_VALUES",
                            "characteristic": characteristic_name,
                            "state": str(state),
                            "feature_index": feature_idx,
                            "feature_name": self._feature_name(feature_idx),
                            "mean": self._to_serializable(stats["mean"]),
                            "std": self._to_serializable(stats["std"]),
                            "sum": "",
                            "min": self._to_serializable(stats["min"]),
                            "max": self._to_serializable(stats["max"]),
                            "count": stats["count"],
                        }
                    )

        # --------------------------------------------------
        # write csv
        if output_dir is not None:
            output_path = Path(output_dir).parent
            output_path.mkdir(parents=True, exist_ok=True)

        with open(output_dir, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "summary_type",
                    "value_name",
                    "characteristic",
                    "state",
                    "feature_index",
                    "feature_name",
                    "mean",
                    "std",
                    "sum",
                    "min",
                    "max",
                    "count",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    # ------------------------------------------------------------------
    # Flattening

    def flatten_results(self) -> List[Dict[str, Any]]:
        """Flatten all results into row dictionaries."""
        rows = []

        for value_name, characteristic_dict in self.results.items():
            for characteristic_name, values in characteristic_dict.items():
                for state, feature_values in values.items():
                    for feature_idx, value in enumerate(feature_values):
                        rows.append(
                            {
                                "value_name": value_name,
                                "characteristic": characteristic_name,
                                "state": str(tuple(state)),
                                "feature_index": feature_idx,
                                "feature_name": self._feature_name(feature_idx),
                                "value": self._to_serializable(value),
                            }
                        )

        return rows

    # ------------------------------------------------------------------
    # Helpers

    def format_value(self, value: Any) -> str:
        """Format scalar or vector values for terminal output."""
        arr = np.asarray(value)

        if arr.ndim == 0:
            return f"{float(arr):.{self.digits}f}"

        flat = arr.flatten()
        return "[" + ", ".join(f"{float(v):.{self.digits}f}" for v in flat) + "]"

    def _feature_name(self, idx: int) -> str:
        if self.feature_names is not None and idx < len(self.feature_names):
            return self.feature_names[idx]
        return f"Feature {idx}"

    def _normalize_state(self, state: Any) -> State:
        if isinstance(state, tuple):
            return state
        if isinstance(state, np.ndarray):
            return tuple(state.tolist())
        if isinstance(state, list):
            return tuple(state)
        return (state,)

    def _group_by_feature(self, values: StateValues) -> Dict[int, List[Any]]:
        grouped: Dict[int, List[Any]] = {}

        for _state, feature_values in values.items():
            for feature_idx, value in enumerate(feature_values):
                grouped.setdefault(feature_idx, []).append(value)

        return grouped

    def _stack_values(self, values: Iterable[Any]) -> np.ndarray:
        arrays = [np.asarray(value, dtype=float) for value in values]
        return np.stack(arrays, axis=0)

    def _to_serializable(self, value: Any) -> Any:
        arr = np.asarray(value)

        if arr.ndim == 0:
            return float(arr)

        return arr.tolist()

    def _jsonify(self, obj: Any) -> Any:
        """Recursively convert tuple keys and numpy values to JSON-safe objects."""
        if isinstance(obj, dict):
            return {str(k): self._jsonify(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple)):
            return [self._jsonify(v) for v in obj]

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()

        return obj
