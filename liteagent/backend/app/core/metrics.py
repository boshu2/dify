"""
Simple in-memory metrics collector.

For production use, consider using prometheus-client directly.
"""
import threading
from typing import Any


class MetricsCollector:
    """Simple in-memory metrics collector."""

    def __init__(self):
        self._counters: dict[str, int] = {}
        self._histograms: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def increment(
        self,
        name: str,
        value: int = 1,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + value

    def get_counter(self, name: str, labels: dict[str, str] | None = None) -> int:
        """Get the current value of a counter."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._counters.get(key, 0)

    def record_histogram(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record a value in a histogram metric."""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)

    def get_histogram(
        self, name: str, labels: dict[str, str] | None = None
    ) -> list[float] | None:
        """Get histogram values."""
        key = self._make_key(name, labels)
        with self._lock:
            return self._histograms.get(key)

    def _make_key(self, name: str, labels: dict[str, str] | None) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


# Global metrics instance
metrics = MetricsCollector()
