"""
Caching and Telemetry System for LangChain RAG Performance Monitoring
Adapted from original RAG implementation
"""

from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Dict, List, Optional

from langchain_core.documents import Document
import structlog

logger = structlog.get_logger(__name__)


class QueryCache:
    """Cache query embeddings and retrieval results for LangChain Documents."""

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        ttl_seconds: int = 3600,
        max_memory_entries: int = 1000,
        max_disk_size_mb: int = 500,
        auto_cleanup_interval: int = 100,
    ):
        self.logger = logger.bind(component="QueryCache")
        self.cache_dir = Path(cache_dir or "data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.max_memory_entries = max_memory_entries
        self.max_disk_size_mb = max_disk_size_mb
        self.auto_cleanup_interval = auto_cleanup_interval

        # In-memory cache for fast access (with access tracking for LRU)
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.access_order: List[str] = []  # Track access order for LRU
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "cleanups": 0,
            "operations": 0,
        }

        # Run initial cleanup
        self.cleanup_expired_files()

    def _get_cache_key(self, query: str, context: Optional[str] = None) -> str:
        """Generate cache key from query and optional context."""
        content = f"{query}:{context or ''}"
        content = content.replace("\n", "").replace("\r", "").replace("\t", "")
        content = content.replace("?", "").replace("!", "").replace("`", "")
        content = content.replace(" ", "").replace(".", "").replace(",", "").strip()
        return hashlib.md5(content.encode()).hexdigest()

    def get(
        self, query: str, context: Optional[str] = None
    ) -> Optional[List[Document]]:
        """Get cached result if available and not expired."""
        cache_key = self._get_cache_key(query, context)

        # Check memory cache first
        if cache_key in self.memory_cache:
            entry = self.memory_cache[cache_key]

            # Check TTL
            if time.time() - entry["timestamp"] < self.ttl_seconds:
                self.cache_stats["hits"] += 1
                self._update_access_order(cache_key)
                self.logger.debug(f"✅ Cache hit for query: {query[:50]}...")
                return entry["data"]
            else:
                # Expired - remove from memory
                del self.memory_cache[cache_key]
                if cache_key in self.access_order:
                    self.access_order.remove(cache_key)

        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    entry = pickle.load(f)

                # Check TTL
                if time.time() - entry["timestamp"] < self.ttl_seconds:
                    # Load into memory cache (with LRU eviction if needed)
                    self._add_to_memory_cache(cache_key, entry)
                    self.cache_stats["hits"] += 1
                    self.logger.debug(f"✅ Cache hit (disk) for query: {query[:50]}...")
                    return entry["data"]
                else:
                    # Remove expired cache file
                    cache_file.unlink()
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load cache: {e}")

        self.cache_stats["misses"] += 1
        return None

    def set(self, query: str, data: List[Document], context: Optional[str] = None):
        """Cache query result."""
        cache_key = self._get_cache_key(query, context)

        entry = {"data": data, "timestamp": time.time(), "query": query}

        # Store in memory cache (with LRU eviction if needed)
        self._add_to_memory_cache(cache_key, entry)

        # Persist to disk
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, "wb") as f:
                pickle.dump(entry, f)
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to persist cache: {e}")

        # Increment operations counter and trigger periodic cleanup
        self.cache_stats["operations"] += 1
        if self.cache_stats["operations"] % self.auto_cleanup_interval == 0:
            self._auto_cleanup()

    def clear(self):
        """Clear all caches."""
        self.memory_cache.clear()
        self.access_order.clear()

        # Clear disk cache
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except Exception:
                pass

        self.logger.info("🗑️ Cache cleared")

    def _update_access_order(self, cache_key: str):
        """Update access order for LRU (move to end as most recently used)."""
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)

    def _add_to_memory_cache(self, cache_key: str, entry: Dict[str, Any]):
        """Add entry to memory cache with LRU eviction if needed."""
        # Check if we need to evict
        while len(self.memory_cache) >= self.max_memory_entries:
            # Evict least recently used
            if self.access_order:
                lru_key = self.access_order.pop(0)
                if lru_key in self.memory_cache:
                    del self.memory_cache[lru_key]
                    self.cache_stats["evictions"] += 1
            else:
                # Fallback: remove arbitrary item
                arbitrary_key = next(iter(self.memory_cache))
                del self.memory_cache[arbitrary_key]
                self.cache_stats["evictions"] += 1

        # Add to cache
        self.memory_cache[cache_key] = entry
        self._update_access_order(cache_key)

    def cleanup_expired_files(self) -> int:
        """Remove expired cache files from disk. Returns number of files removed."""
        removed_count = 0
        current_time = time.time()

        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    # Load and check expiration
                    with open(cache_file, "rb") as f:
                        entry = pickle.load(f)

                    if current_time - entry["timestamp"] >= self.ttl_seconds:
                        cache_file.unlink()
                        removed_count += 1
                except Exception as e:
                    # If we can't read the file, consider it corrupt and remove it
                    self.logger.warning(
                        f"Removing corrupt cache file {cache_file.name}: {e}"
                    )
                    try:
                        cache_file.unlink()
                        removed_count += 1
                    except Exception:
                        pass

            if removed_count > 0:
                self.logger.info(f"🧹 Cleaned up {removed_count} expired cache files")
        except Exception as e:
            self.logger.warning(f"⚠️ Error during cache cleanup: {e}")

        return removed_count

    def enforce_disk_size_limit(self) -> int:
        """Enforce maximum disk cache size by removing oldest files. Returns number of files removed."""
        removed_count = 0

        try:
            # Get all cache files with their sizes and timestamps
            cache_files = []
            total_size_mb = 0

            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    file_size = cache_file.stat().st_size / (
                        1024 * 1024
                    )  # Convert to MB
                    total_size_mb += file_size

                    # Try to get timestamp from file content
                    with open(cache_file, "rb") as f:
                        entry = pickle.load(f)
                    timestamp = entry.get("timestamp", cache_file.stat().st_mtime)

                    cache_files.append(
                        {
                            "path": cache_file,
                            "size_mb": file_size,
                            "timestamp": timestamp,
                        }
                    )
                except Exception:
                    # Skip files we can't read
                    continue

            # If over limit, remove oldest files
            if total_size_mb > self.max_disk_size_mb:
                # Sort by timestamp (oldest first)
                cache_files.sort(key=lambda x: x["timestamp"])

                # Remove oldest files until under limit
                for file_info in cache_files:
                    if total_size_mb <= self.max_disk_size_mb:
                        break

                    try:
                        file_info["path"].unlink()
                        total_size_mb -= file_info["size_mb"]
                        removed_count += 1
                    except Exception:
                        pass

                if removed_count > 0:
                    self.logger.info(
                        f"🧹 Removed {removed_count} old cache files to enforce size limit "
                        f"(now {total_size_mb:.2f}MB / {self.max_disk_size_mb}MB)"
                    )
        except Exception as e:
            self.logger.warning(f"⚠️ Error enforcing disk size limit: {e}")

        return removed_count

    def _auto_cleanup(self):
        """Automatic cleanup triggered periodically."""
        try:
            expired = self.cleanup_expired_files()
            size_limit = self.enforce_disk_size_limit()

            if expired > 0 or size_limit > 0:
                self.cache_stats["cleanups"] += 1
                self.logger.debug(
                    f"Auto cleanup: {expired} expired, {size_limit} for size limit"
                )
        except Exception as e:
            self.logger.warning(f"⚠️ Auto cleanup failed: {e}")

    def get_cache_size_info(self) -> Dict[str, Any]:
        """Get information about cache disk usage."""
        total_size_bytes = 0
        file_count = 0

        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                try:
                    total_size_bytes += cache_file.stat().st_size
                    file_count += 1
                except Exception:
                    pass
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting cache size: {e}")

        return {
            "total_size_mb": total_size_bytes / (1024 * 1024),
            "total_size_bytes": total_size_bytes,
            "file_count": file_count,
            "max_size_mb": self.max_disk_size_mb,
            "utilization": (
                (total_size_bytes / (1024 * 1024)) / self.max_disk_size_mb
                if self.max_disk_size_mb > 0
                else 0
            ),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (
            self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        )

        cache_size_info = self.get_cache_size_info()

        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.memory_cache),
            "evictions": self.cache_stats["evictions"],
            "cleanups": self.cache_stats["cleanups"],
            "operations": self.cache_stats["operations"],
            "disk_cache_size_mb": cache_size_info["total_size_mb"],
            "disk_file_count": cache_size_info["file_count"],
            "disk_utilization": cache_size_info["utilization"],
        }


class PerformanceTelemetry:
    """Track and log performance metrics for LangChain RAG system."""

    def __init__(self, log_file: Optional[str] = None):
        self.logger = logger.bind(component="PerformanceTelemetry")
        self.log_file = Path(log_file or "logs/monitoring_metrics.json")
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # In-memory metrics
        self.metrics = defaultdict(list)
        self.stage_times = defaultdict(list)
        self.current_query_id = None
        self.current_stages = {}

    def start_query(self, query: str) -> str:
        """Start tracking a new query."""
        query_id = hashlib.md5(f"{query}{time.time()}".encode()).hexdigest()[:16]
        self.current_query_id = query_id
        self.current_stages = {
            "query_id": query_id,
            "query": query,
            "start_time": time.time(),
            "stages": {},
        }
        return query_id

    def start_stage(self, stage_name: str):
        """Start timing a stage."""
        if self.current_query_id:
            self.current_stages["stages"][stage_name] = {"start_time": time.time()}

    def end_stage(self, stage_name: str, metadata: Optional[Dict[str, Any]] = None):
        """End timing a stage and record metrics."""
        if self.current_query_id and stage_name in self.current_stages["stages"]:
            stage = self.current_stages["stages"][stage_name]
            elapsed = time.time() - stage["start_time"]

            stage["elapsed_ms"] = elapsed * 1000
            stage["metadata"] = metadata or {}

            # Record in aggregated metrics
            self.stage_times[stage_name].append(elapsed)

    def record_retrieval_metrics(
        self, stage_name: str, results_count: int, recall_at_k: Optional[float] = None
    ):
        """Record retrieval-specific metrics."""
        metrics = {"results_count": results_count, "recall_at_k": recall_at_k}

        if self.current_query_id and stage_name in self.current_stages["stages"]:
            self.current_stages["stages"][stage_name]["retrieval_metrics"] = metrics

    def end_query(self, results_count: int, success: bool = True):
        """End query tracking and log metrics."""
        if not self.current_query_id:
            return

        total_time = time.time() - self.current_stages["start_time"]

        self.current_stages.update(
            {
                "total_time_ms": total_time * 1000,
                "results_count": results_count,
                "success": success,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Log to file
        self._log_to_file(self.current_stages)

        # Store in memory for aggregation
        self.metrics["queries"].append(self.current_stages)

        # Reset
        self.current_query_id = None
        self.current_stages = {}

    def _log_to_file(self, query_metrics: Dict[str, Any]):
        """Append metrics to log file."""
        try:
            # Append to JSONL file
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(query_metrics) + "\n")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to log metrics: {e}")

    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated performance metrics."""
        import numpy as np

        metrics = {}

        # Overall metrics
        total_queries = len(self.metrics["queries"])
        if total_queries > 0:
            metrics["total_queries"] = total_queries

            # Success rate
            successful = sum(
                1 for q in self.metrics["queries"] if q.get("success", True)
            )
            metrics["success_rate"] = successful / total_queries

            # Latency statistics
            latencies = [q["total_time_ms"] for q in self.metrics["queries"]]
            metrics["latency"] = {
                "mean_ms": np.mean(latencies),
                "median_ms": np.median(latencies),
                "p95_ms": np.percentile(latencies, 95),
                "p99_ms": np.percentile(latencies, 99),
                "max_ms": np.max(latencies),
                "min_ms": np.min(latencies),
            }

        # Stage-specific metrics
        metrics["stages"] = {}
        for stage_name, times in self.stage_times.items():
            if times:
                metrics["stages"][stage_name] = {
                    "mean_ms": np.mean(times) * 1000,
                    "median_ms": np.median(times) * 1000,
                    "p95_ms": np.percentile(times, 95) * 1000,
                    "count": len(times),
                }

        return metrics

    def print_report(self):
        """Print a formatted performance report."""
        metrics = self.get_aggregated_metrics()

        print("\n" + "=" * 80)
        print("LANGCHAIN RAG PERFORMANCE REPORT")
        print("=" * 80)

        if "total_queries" in metrics:
            print("\n📊 Overall Metrics:")
            print(f"  Total Queries: {metrics['total_queries']}")
            print(f"  Success Rate: {metrics['success_rate']:.2%}")

            if "latency" in metrics:
                print("\n⏱️  Latency Statistics:")
                lat = metrics["latency"]
                print(f"  Mean: {lat['mean_ms']:.2f}ms")
                print(f"  Median: {lat['median_ms']:.2f}ms")
                print(f"  P95: {lat['p95_ms']:.2f}ms")
                print(f"  P99: {lat['p99_ms']:.2f}ms")

        if "stages" in metrics and metrics["stages"]:
            print("\n🔄 Stage Breakdown:")
            for stage_name, stage_metrics in metrics["stages"].items():
                print(f"\n  {stage_name}:")
                print(f"    Mean: {stage_metrics['mean_ms']:.2f}ms")
                print(f"    Median: {stage_metrics['median_ms']:.2f}ms")
                print(f"    P95: {stage_metrics['p95_ms']:.2f}ms")
                print(f"    Count: {stage_metrics['count']}")

        print("\n" + "=" * 80 + "\n")


class CachedRetriever:
    """Wrapper that adds caching to LangChain hybrid retriever."""

    def __init__(self, retriever: Any, cache: Optional[QueryCache] = None):
        self.retriever = retriever
        self.cache = cache or QueryCache()
        self.logger = logger.bind(component="CachedRetriever")

    def retrieve(
        self, query: str, fetch_k: int = 10, filter_types: List[str] = None, **kwargs
    ) -> List[Document]:
        """Retrieve with caching."""
        # Create cache context from kwargs and filter_types
        context_parts = [f"k={fetch_k}"]
        if filter_types:
            context_parts.append(f"types={','.join(sorted(filter_types))}")
        context_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        context = ":".join(context_parts)

        # Try cache first
        cached_results = self.cache.get(query, context)
        if cached_results is not None:
            return cached_results

        # Cache miss - perform actual retrieval
        results = self.retriever.retrieve(query, fetch_k, filter_types, **kwargs)

        # Cache results
        self.cache.set(query, results, context)

        return results

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()


class InstrumentedRetriever:
    """Wrapper that adds telemetry to LangChain hybrid retriever."""

    def __init__(
        self,
        retriever: Any,
        telemetry: Optional[PerformanceTelemetry] = None,
        stage_name: str = "retrieval",
    ):
        self.retriever = retriever
        self.telemetry = telemetry or PerformanceTelemetry()
        self.stage_name = stage_name
        self.logger = logger.bind(component="InstrumentedRetriever")

    def retrieve(
        self, query: str, fetch_k: int = 10, filter_types: List[str] = None, **kwargs
    ) -> List[Document]:
        """Retrieve with telemetry."""
        # Start tracking
        self.telemetry.start_stage(self.stage_name)

        try:
            # Perform retrieval
            results = self.retriever.retrieve(query, fetch_k, filter_types, **kwargs)

            # Record metrics
            metadata = {"fetch_k": fetch_k, **kwargs}
            if filter_types:
                metadata["filter_types"] = filter_types

            self.telemetry.record_retrieval_metrics(
                self.stage_name, results_count=len(results)
            )

            return results

        finally:
            # End tracking
            metadata = {"fetch_k": fetch_k, **kwargs}
            if filter_types:
                metadata["filter_types"] = filter_types
            self.telemetry.end_stage(self.stage_name, metadata)

    def get_telemetry_report(self) -> Dict[str, Any]:
        """Get telemetry report."""
        return self.telemetry.get_aggregated_metrics()


def cached_function(cache: Optional[QueryCache] = None, ttl: Optional[int] = None):
    """Decorator to cache function results."""
    if cache is None:
        cache = QueryCache(ttl_seconds=ttl or 3600)

    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)

            # Try cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Call function
            result = func(*args, **kwargs)

            # Cache result
            cache.set(cache_key, result)

            return result

        return wrapper

    return decorator
