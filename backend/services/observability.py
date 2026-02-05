"""
Module: observability.py
Description: Observability, logging, and metrics tracking for Smart Financial Coach.

Features:
    - Structured logging with context
    - Timing decorators for performance monitoring
    - Metrics collection and reporting
    - Request tracking

Usage:
    from services.observability import logger, metrics, timed

    @timed("categorize_transactions")
    async def categorize(transactions):
        logger.info("Categorizing", count=len(transactions))
        ...

Author: Smart Financial Coach Team
Created: 2025-01-31
"""

import time
import logging
import functools
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from collections import defaultdict
from contextlib import contextmanager
import json


# =============================================================================
# Structured Logger
# =============================================================================

class StructuredLogger:
    """
    Structured logger with JSON-compatible output and context tracking.
    
    Provides:
        - Contextual logging (session_id, request_id)
        - JSON-formatted output for log aggregation
        - Log level management
        - Performance tracking
    """
    
    def __init__(self, name: str = "smart-financial-coach"):
        """Initialize logger with given name."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Add handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs) -> None:
        """Set context fields that will be included in all logs."""
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear all context fields."""
        self._context = {}
    
    def _format_message(self, message: str, **kwargs) -> str:
        """Format message with context and additional fields."""
        fields = {**self._context, **kwargs}
        if fields:
            field_str = " | ".join(f"{k}={v}" for k, v in fields.items())
            return f"{message} | {field_str}"
        return message
    
    def info(self, message: str, **kwargs) -> None:
        """Log info level message."""
        self.logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning level message."""
        self.logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs) -> None:
        """Log error level message."""
        self.logger.error(self._format_message(message, **kwargs))
    
    def debug(self, message: str, **kwargs) -> None:
        """Log debug level message."""
        self.logger.debug(self._format_message(message, **kwargs))
    
    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(self._format_message(message, **kwargs))


# =============================================================================
# Metrics Collector
# =============================================================================

class MetricsCollector:
    """
    Simple in-memory metrics collection for monitoring and debugging.
    
    Collects:
        - Counters (requests, errors, anomalies)
        - Gauges (current values)
        - Histograms (timing distributions)
        - Per-session metrics
    
    Note: In production, replace with Prometheus/StatsD/DataDog client.
    """
    
    def __init__(self):
        """Initialize metrics storage."""
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.timings: Dict[str, list] = defaultdict(list)
        self.session_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._start_time = datetime.utcnow()
    
    def increment(self, name: str, value: int = 1, tags: Dict[str, str] = None) -> None:
        """Increment a counter."""
        key = self._make_key(name, tags)
        self.counters[key] += value
    
    def gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Set a gauge value."""
        key = self._make_key(name, tags)
        self.gauges[key] = value
    
    def timing(self, name: str, duration_ms: float, tags: Dict[str, str] = None) -> None:
        """Record a timing measurement."""
        key = self._make_key(name, tags)
        self.timings[key].append(duration_ms)
        # Keep only last 1000 measurements
        if len(self.timings[key]) > 1000:
            self.timings[key] = self.timings[key][-1000:]
    
    def record_session(self, session_id: str, **kwargs) -> None:
        """Record session-specific metrics."""
        self.session_metrics[session_id].update(kwargs)
    
    def _make_key(self, name: str, tags: Optional[Dict[str, str]] = None) -> str:
        """Create metric key with optional tags."""
        if tags:
            tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
            return f"{name}:{tag_str}"
        return name
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        summary = {
            "uptime_seconds": (datetime.utcnow() - self._start_time).total_seconds(),
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "timings": {},
            "session_count": len(self.session_metrics),
        }
        
        # Calculate timing statistics
        for name, values in self.timings.items():
            if values:
                summary["timings"][name] = {
                    "count": len(values),
                    "avg_ms": sum(values) / len(values),
                    "min_ms": min(values),
                    "max_ms": max(values),
                    "p50_ms": sorted(values)[len(values) // 2],
                    "p95_ms": sorted(values)[int(len(values) * 0.95)] if len(values) >= 20 else None,
                }
        
        return summary
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get metrics for a specific session."""
        return dict(self.session_metrics.get(session_id, {}))


# =============================================================================
# Timing Decorators
# =============================================================================

def timed(name: str = None):
    """
    Decorator to time function execution and record metrics.
    
    Args:
        name: Metric name (defaults to function name).
        
    Example:
        @timed("anomaly_detection")
        def detect_anomalies(transactions):
            ...
    """
    def decorator(func: Callable) -> Callable:
        metric_name = name or func.__name__
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                metrics.increment(f"{metric_name}.success")
                return result
            except Exception as e:
                metrics.increment(f"{metric_name}.error")
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                metrics.timing(metric_name, duration_ms)
                logger.debug(f"{metric_name} completed", duration_ms=f"{duration_ms:.2f}")
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                metrics.increment(f"{metric_name}.success")
                return result
            except Exception as e:
                metrics.increment(f"{metric_name}.error")
                raise
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                metrics.timing(metric_name, duration_ms)
                logger.debug(f"{metric_name} completed", duration_ms=f"{duration_ms:.2f}")
        
        # Return appropriate wrapper based on function type
        if asyncio_iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def asyncio_iscoroutinefunction(func: Callable) -> bool:
    """Check if function is async."""
    import asyncio
    return asyncio.iscoroutinefunction(func)


@contextmanager
def timed_block(name: str):
    """
    Context manager for timing code blocks.
    
    Example:
        with timed_block("data_processing"):
            process_data()
    """
    start = time.perf_counter()
    try:
        yield
        metrics.increment(f"{name}.success")
    except Exception:
        metrics.increment(f"{name}.error")
        raise
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        metrics.timing(name, duration_ms)


# =============================================================================
# Global Instances
# =============================================================================

# Global logger instance
logger = StructuredLogger()

# Global metrics collector
metrics = MetricsCollector()


# =============================================================================
# Convenience Functions
# =============================================================================

def log_analysis_start(session_id: str, transaction_count: int) -> None:
    """Log the start of an analysis."""
    logger.set_context(session_id=session_id[:8])
    logger.info("Analysis started", transactions=transaction_count)
    metrics.increment("analysis.started")
    metrics.record_session(session_id, 
                          transaction_count=transaction_count,
                          start_time=datetime.utcnow().isoformat())


def log_analysis_complete(session_id: str, results: Dict[str, int]) -> None:
    """Log completion of an analysis."""
    logger.info("Analysis completed", **results)
    metrics.increment("analysis.completed")
    metrics.record_session(session_id, 
                          results=results,
                          end_time=datetime.utcnow().isoformat())
    logger.clear_context()


def log_anomaly_detected(session_id: str, severity: str, amount: float) -> None:
    """Log an anomaly detection."""
    logger.info("Anomaly detected", severity=severity, amount=f"${amount:.2f}")
    metrics.increment("anomalies.detected", tags={"severity": severity})


def log_chat_request(session_id: str, message_length: int) -> None:
    """Log a chat request."""
    logger.info("Chat request", msg_length=message_length)
    metrics.increment("chat.requests")


def log_openai_call(endpoint: str, tokens: int, duration_ms: float) -> None:
    """Log an OpenAI API call."""
    logger.debug("OpenAI API call", endpoint=endpoint, tokens=tokens, duration_ms=f"{duration_ms:.2f}")
    metrics.increment("openai.calls")
    metrics.increment("openai.tokens", tokens)
    metrics.timing("openai.latency", duration_ms)
