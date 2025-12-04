"""
DARA Services - Inference Cache
Provides LRU caching for model inference results to speed up repeated queries.
"""

from typing import Optional, Any
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import time

from ..utils.logging import get_logger

logger = get_logger("cache")


@dataclass
class CacheEntry:
    """Single cache entry with metadata."""
    value: Any
    timestamp: float
    hits: int = 0
    
    def touch(self) -> None:
        """Update access metadata."""
        self.hits += 1
        self.timestamp = time.time()


class InferenceCache:
    """
    LRU cache for model inference results.
    
    Features:
    - In-memory LRU eviction
    - Optional disk persistence
    - Cache statistics
    """
    
    def __init__(
        self,
        maxsize: int = 100,
        persist_path: Optional[str] = None,
        ttl_seconds: Optional[int] = None
    ):
        """
        Initialize the cache.
        
        Args:
            maxsize: Maximum number of entries
            persist_path: Optional path for disk persistence
            ttl_seconds: Optional time-to-live for entries
        """
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self.persist_path = Path(persist_path) if persist_path else None
        
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}
        
        if self.persist_path and self.persist_path.exists():
            self._load_from_disk()
    
    def _make_key(self, image_hash: str, prompt: str, **kwargs) -> str:
        """Create cache key from inputs."""
        key_data = f"{image_hash}:{prompt}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    def get(self, image_hash: str, prompt: str, **kwargs) -> Optional[Any]:
        """
        Get cached result if available.
        
        Args:
            image_hash: Hash of input image
            prompt: Model prompt used
            **kwargs: Additional parameters
            
        Returns:
            Cached result or None
        """
        key = self._make_key(image_hash, prompt, **kwargs)
        
        if key not in self._cache:
            self._stats["misses"] += 1
            return None
        
        entry = self._cache[key]
        
        # Check TTL
        if self.ttl_seconds and (time.time() - entry.timestamp) > self.ttl_seconds:
            del self._cache[key]
            self._stats["misses"] += 1
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        entry.touch()
        self._stats["hits"] += 1
        
        logger.debug(f"Cache hit for key {key[:8]}... (hits: {entry.hits})")
        return entry.value
    
    def set(self, image_hash: str, prompt: str, result: Any, **kwargs) -> None:
        """
        Store result in cache.
        
        Args:
            image_hash: Hash of input image
            prompt: Model prompt used
            result: Result to cache
            **kwargs: Additional parameters
        """
        key = self._make_key(image_hash, prompt, **kwargs)
        
        # Evict if at capacity
        while len(self._cache) >= self.maxsize:
            evicted_key, _ = self._cache.popitem(last=False)
            self._stats["evictions"] += 1
            logger.debug(f"Evicted cache entry {evicted_key[:8]}...")
        
        self._cache[key] = CacheEntry(value=result, timestamp=time.time())
        self._cache.move_to_end(key)
        
        logger.debug(f"Cached result for key {key[:8]}...")
    
    def clear(self) -> int:
        """Clear all cache entries. Returns count of cleared entries."""
        count = len(self._cache)
        self._cache.clear()
        logger.info(f"Cleared {count} cache entries")
        return count
    
    @property
    def size(self) -> int:
        """Current number of entries."""
        return len(self._cache)
    
    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {
            **self._stats,
            "size": self.size,
            "maxsize": self.maxsize,
            "hit_rate": round(hit_rate, 3)
        }
    
    def _save_to_disk(self) -> None:
        """Persist cache to disk."""
        if not self.persist_path:
            return
        
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            # Only save string-serializable entries
            data = {
                k: {"value": v.value, "timestamp": v.timestamp}
                for k, v in self._cache.items()
                if isinstance(v.value, (str, dict, list))
            }
            with open(self.persist_path, "w") as f:
                json.dump(data, f)
            logger.info(f"Saved {len(data)} cache entries to disk")
        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")
    
    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            
            for key, entry in data.items():
                self._cache[key] = CacheEntry(
                    value=entry["value"],
                    timestamp=entry["timestamp"]
                )
            logger.info(f"Loaded {len(data)} cache entries from disk")
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")
    
    def __del__(self):
        """Save to disk on cleanup."""
        if self.persist_path:
            self._save_to_disk()
