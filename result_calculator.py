"""
result_calculator.py

Shared calculation and formatting utilities for result_analyzer.py and
result_visualizer.py.

This class contains only data normalization, numerical calculations,
comparisons, matrix conversion, and serialization helpers. It does not print,
export files, or create plots.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

State = Tuple[Any, ...]
FeatureValue = Union[float, int, np.ndarray, List[float], Tuple[float, ...]]
StateValues = Mapping[State, List[FeatureValue]]
Results = Mapping[str, Mapping[str, StateValues]]


class ResultCalculator:
    """
    Shared calculation class for nested explanation results.

    Expected strtucture:
        results[value_name][characteristic_name][state][feature_index] = value

    Example:
        calculator = ResultCalculator(results, feature_names=["x", "y"])
        summary = calculator.per_feature_summary()
        diff = calculator.compare_values("tau", "shapley")
        states, matrix = calculator.values_to_matrix("tau", "local")
    """

    def __init__(
        self,
        results: Optional[Results] = None,
        feature_names: Optional[List[str]] = None,
        digits: int = 3,
    ) -> None:
        self.results: Dict[str, Dict[str, Dict[State, List[Any]]]] = {}
        self.feature_names = feature_names
        self.digits = digits

        if results is not None:
            self.add_results(results)

    # ------------------------------------------------------------------
    # Data management

    def add_result(self, value_name: str, characteristic_name: str, values: StateValues) -> None:
        """Add one result table and normalize its state keys."""
        self.results.setdefault(value_name, {})[characteristic_name] = {
            self.normalize_state(state): list(feature_values)
            for state, feature_values in values.items()
        }

    def add_results(self, results: Results) -> None:
        """Add many nested results at once."""
        for value_name, characteristic_dict in results.items():
            for characteristic_name, values in characteristic_dict.items():
                self.add_result(value_name, characteristic_name, values)

    # ------------------------------------------------------------------
    # Summaries

    def per_feature_summary(self) -> Dict[str, Dict[str, Dict[int, Dict[str, Any]]]]:
        """
        Compute mean, std, min, max and count per value type,
        characteristic and feature across all states.
        """
        output: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}

        for value_name, characteristic_dict in self.results.items():
            output[value_name] = {}

            for characteristic_name, values in characteristic_dict.items():
                grouped = self.group_by_feature(values)
                output[value_name][characteristic_name] = {}

                for feature_idx, feature_values in grouped.items():
                    arr = self.stack_values(feature_values)
                    output[value_name][characteristic_name][feature_idx] = self.basic_stats(arr)

        return output

    def per_state_summary(self) -> Dict[str, Dict[str, Dict[State, Dict[str, Any]]]]:
        """Compute summary statistics across features for each state."""
        output: Dict[str, Dict[str, Dict[State, Dict[str, Any]]]] = {}

        for value_name, characteristic_dict in self.results.items():
            output[value_name] = {}

            for characteristic_name, values in characteristic_dict.items():
                output[value_name][characteristic_name] = {}

                for state, feature_values in values.items():
                    arr = self.stack_values(feature_values)
                    output[value_name][characteristic_name][state] = {
                        "mean_across_features": np.mean(arr, axis=0),
                        "std_across_features":np.std(arr, axis=0),
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
        """
        grouped: Dict[str, Dict[State, Dict[int, List[Any]]]] = {}

        for _value_name, characteristic_dict in self.results.items():
            for characteristic_name, values in characteristic_dict.items():
                grouped.setdefault(characteristic_name, {})

                for state, feature_values in values.items():
                    state = self.normalize_state(state)
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
                    arr = self.stack_values(values)
                    output[characteristic_name][state][feature_idx] = self.basic_stats(arr)

        return output

    def basic_stats(self, arr: np.ndarray) -> Dict[str, Any]:
        """Compute basic statistics along axis 0."""
        return {
            "mean": np.mean(arr, axis=0),
            "std": np.std(arr, axis=0),
            "min": np.min(arr, axis=0),
            "max": np.max(arr, axis=0),
            "count": int(arr.shape[0]),
        }

    # ------------------------------------------------------------------
    # Comparisons

    def compare_values(
        self,
        left_value: str,
        right_value: str,
        characteristic_names: Optional[Sequence[str]] = None,
        absolute: bool = False,
    ) -> Dict[str, Dict[State, List[Any]]]:
        """
        Compare two value types by computing left_value - right_value.

        If absolute=True, computes abs(left_value - right_value).
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
            raise ValueError(f"No common characteristics found between {left_value} and {right_value}.")

        output: Dict[str, Dict[State, List[Any]]] = {}

        for characteristic_name in common_chars:
            output[characteristic_name] = {}

            left_values = self.results[left_value][characteristic_name]
            right_values = self.results[right_value][characteristic_name]

            common_states = sorted(set(left_values.keys()) & set(right_values.keys()), key=str)

            for state in common_states:
                left_feature_values = left_values[state]
                right_feature_values = right_values[state]

                if len(left_feature_values) != len(right_feature_values):
                    raise ValueError(
                        f"Feature count mismatch for {characteristic_name}, state {state}."
                    )

                diffs = []
                for left, right in zip(left_feature_values, right_feature_values):
                    diff = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
                    if absolute:
                        diff = np.abs(diff)
                    diff = self.clean_small_values(diff)
                    diffs.append(self.pythonify(diff))

                output[characteristic_name][state] = diffs

        return output

    # ------------------------------------------------------------------
    # Matrix conversion for visualizations

    def values_to_matrix(
        self,
        value_name: str,
        characteristic_name: str,
        component: Optional[int] = None,
        vector_mode: str = "mean",
    ) -> Tuple[List[State], np.ndarray]:
        """Convert one result table to a 2D matrix with rows=states, columns=features."""
        if value_name not in self.results:
            raise KeyError(f"Unknown value type: {value_name}")
        if characteristic_name not in self.results[value_name]:
            raise KeyError(f"Unknown characteristic '{characteristic_name}' for value type '{value_name}'")

        return self.state_values_to_matrix(
            self.results[value_name][characteristic_name],
            component=component,
            vector_mode=vector_mode,
        )

    def state_values_to_matrix(
        self,
        values: Mapping[State, List[Any]],
        component: Optional[int] = None,
        vector_mode: str = "mean",
    ) -> Tuple[List[State], np.ndarray]:
        """Convert state -> feature values into a 2D matrix."""
        normalized_values = {self.normalize_state(k): v for k, v in values.items()}
        states = sorted(normalized_values.keys(), key=str)
        rows = []

        for state in states:
            feature_values = normalized_values[state]
            row = [
                self.scalarize_value(value, component=component, vector_mode=vector_mode)
                for value in feature_values
            ]
            rows.append(row)

        return states, self.clean_small_values(np.asarray(rows, dtype=float))

    def difference_matrix(
        self,
        left_value: str,
        right_value: str,
        characteristic_name: str,
        component: Optional[int] = None,
        vector_mode: str = "mean",
        absolute: bool = False,
    ) -> Tuple[List[State], np.ndarray]:
        """Compute a 2D difference matrix left_value - right_value for one characteristic."""
        states, left_matrix = self.values_to_matrix(
            left_value, characteristic_name, component=component, vector_mode=vector_mode
        )
        right_states, right_matrix = self.values_to_matrix(
            right_value, characteristic_name, component=component, vector_mode=vector_mode
        )

        if states != right_states:
            raise ValueError(
                f"State mismatch for characteristic {characteristic_name}. "
                "Both value types must contain the same states in the same normalized order."
            )

        diff_matrix = left_matrix - right_matrix
        if absolute:
            diff_matrix = np.abs(diff_matrix)

        return states, self.clean_small_values(diff_matrix)

    def scalarize_value(
        self,
        value: Any,
        component: Optional[int] = None,
        vector_mode: str = "mean",
    ) -> float:
        """Convert scalar or vector feature value to one scalar."""
        arr = np.asarray(value, dtype=float)

        if arr.ndim == 0:
            return float(arr)

        flat = arr.flatten()

        if vector_mode == "component":
            if component is None:
                raise ValueError(
                    "Vector-valued entries found. Pass component=<action_index> "
                    "or use vector_mode='mean', 'sum', 'norm', or 'max_abs'."
                )
            if component < 0 or component >= len(flat):
                raise IndexError(f"component={component} out of bounds for vector of length {len(flat)}")
            return float(flat[component])

        if vector_mode == "mean":
            return float(np.mean(flat))
        if vector_mode == "sum":
            return float(np.sum(flat))
        if vector_mode == "norm":
            return float(np.linalg.norm(flat))
        if vector_mode == "max_abs":
            idx = int(np.argmax(np.abs(flat)))
            return float(flat[idx])

        raise ValueError("Unknown vector_mode. Use 'component', 'mean', 'sum', 'norm', or 'max_abs'.")

    # ------------------------------------------------------------------
    # Selection helpers

    def select_value_names(self, value_names: Optional[Sequence[str]]) -> List[str]:
        """Return selected value names, checking that they exist."""
        if value_names is None:
            return list(self.results.keys())

        missing = [name for name in value_names if name not in self.results]
        if missing:
            raise KeyError(f"Unknown value types: {missing}")

        return list(value_names)

    def select_characteristic_names(
        self,
        characteristic_dict: Mapping[str, Mapping[State, List[Any]]],
        characteristic_names: Optional[Sequence[str]],
    ) -> List[str]:
        """Return selected characteristic names, checking that they exist."""
        if characteristic_names is None:
            return list(characteristic_dict.keys())

        missing = [name for name in characteristic_names if name not in characteristic_dict]
        if missing:
            raise KeyError(f"Unknown characteristics: {missing}")

        return list(characteristic_names)

    def common_characteristics(
        self,
        left_value: str,
        right_value: str,
        characteristic_names: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """Return characteristics that exist in both value types."""
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
            raise ValueError(f"No common characteristics found between {left_value} and {right_value}.")

        return common_chars

    # ------------------------------------------------------------------
    # Generic helpers

    def feature_name(self, idx: int) -> str:
        """Return feature name if provided, otherwise Feature <idx>."""
        if self.feature_names is not None and idx < len(self.feature_names):
            return self.feature_names[idx]
        return f"Feature {idx}"

    def normalize_state(self, state: Any) -> State:
        """Normalize state keys and convert numpy integer values to Python ints."""
        if isinstance(state, tuple):
            return tuple(int(x) if isinstance(x, np.integer) else x for x in state)
        if isinstance(state, np.ndarray):
            return tuple(int(x) if isinstance(x, np.integer) else x for x in state.tolist())
        if isinstance(state, list):
            return tuple(int(x) if isinstance(x, np.integer) else x for x in state)
        if isinstance(state, np.integer):
            return (int(state),)
        return (state,)

    def format_state(self, state: State, compact: bool = False) -> str:
        """
        Convert state tuple to clean human-readable string.

        compact=False -> (0, 1)
        compact=True  -> (0_1)
        """
        cleaned = self.normalize_state(state)
        if compact:
            return "(" + "_".join(str(x) for x in cleaned) + ")"
        return str(cleaned)

    def format_value(self, value: Any) -> str:
        """Format scalar or vector values for terminal output."""
        arr = self.clean_small_values(value)

        if np.asarray(arr).ndim == 0:
            return f"{float(arr):.{self.digits}f}"

        flat = np.asarray(arr).flatten()
        return "[" + ", ".join(
            f"{float(v):.{self.digits}f}" for v in flat
        ) + "]"

    def group_by_feature(self, values: StateValues) -> Dict[int, List[Any]]:
        """Group state values by feature index."""
        grouped: Dict[int, List[Any]] = {}
        for _state, feature_values in values.items():
            for feature_idx, value in enumerate(feature_values):
                grouped.setdefault(feature_idx, []).append(value)
        return grouped

    def stack_values(self, values: Iterable[Any]) -> np.ndarray:
        """Stack scalar/vector values along axis 0."""
        arrays = [np.asarray(value, dtype=float) for value in values]
        return np.stack(arrays, axis=0)

    def clean_small_values(self, value: Any, eps: float = 1e-10) -> Any:
        """Set very small floating-point values to exactly zero."""
        arr = np.asarray(value, dtype=float).copy()
        arr[np.abs(arr) < eps] = 0.0
        arr = np.round(arr, self.digits)
        
        if arr.ndim == 0:
            return float(arr)
        return arr

    def contains_vector_values(self, values: Mapping[State, List[Any]]) -> bool:
        """Check whether any feature value is vector-valued."""
        for feature_values in values.values():
            for value in feature_values:
                if np.asarray(value).ndim > 0:
                    return True
        return False

    def to_serializable(self, value: Any) -> Any:
        """Convert numpy values to JSON/CSV-friendly Python values."""
        arr = self.clean_small_values(value)

        if np.asarray(arr).ndim == 0:
            return float(arr)

        return np.asarray(arr).tolist()

    def pythonify(self, obj: Any) -> Any:
        """Recursively convert numpy values to plain Python values."""
        if isinstance(obj, dict):
            return {
                self.normalize_state(k) if isinstance(k, tuple) else k: self.pythonify(v)
                for k, v in obj.items()
            }

        if isinstance(obj, list):
            return [self.pythonify(v) for v in obj]

        if isinstance(obj, tuple):
            return tuple(self.pythonify(v) for v in obj)

        if isinstance(obj, np.ndarray):
            return self.clean_small_values(obj).tolist()

        if isinstance(obj, (np.integer, np.floating)):
            return self.clean_small_values(obj)

        if isinstance(obj, float):
            return self.clean_small_values(obj)

        return obj

    def jsonify(self, obj: Any) -> Any:
        """Recursively convert tuple keys and numpy values to JSON-safe objects."""
        if isinstance(obj, dict):
            return {
                self.format_state(k) if isinstance(k, tuple) else str(k): self.jsonify(v)
                for k, v in obj.items()
            }

        if isinstance(obj, list):
            return [self.jsonify(v) for v in obj]

        if isinstance(obj, tuple):
            return [self.jsonify(v) for v in obj]

        if isinstance(obj, np.ndarray):
            return self.clean_small_values(obj).tolist()

        if isinstance(obj, (np.integer, np.floating)):
            return self.clean_small_values(obj)

        if isinstance(obj, float):
            return self.clean_small_values(obj)

        return obj

    def safe_filename(self, title: str, file_format: str) -> str:
        """Create a filesystem-friendly filename from a title."""
        safe = title.lower()
        for ch in [" ", "-", "/", "\\", ":", "(", ")", "[", "]", ","]:
            safe = safe.replace(ch, "_")
        while "__" in safe:
            safe = safe.replace("__", "_")
        return safe.strip("_") + f".{file_format}"