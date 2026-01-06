"""
Response Caching Module

Provides caching functionality for LLM responses to avoid redundant API calls
and reduce costs. Uses a TTL-based cache with configurable size limits.

Author: ML Engineer
Date: 2025-01-06
"""

import hashlib
import json
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional
from cachetools import TTLCache
from pydantic import BaseModel


class CacheEntry(BaseModel):
    """Represents a cached response entry."""
    key: str
    value: Any
    created_at: datetime
    expires_at: datetime
    hits: int = 0


class ResponseCache:
    """
    LLM Response Cache with TTL support.
    
    Features:
    - In-memory caching with TTL
    - Persistent disk caching (optional)
    - Cache statistics tracking
    - Configurable size limits
    
    Usage:
        cache = ResponseCache(ttl=3600, max_size=1000)
        
        # Check cache
        cached = cache.get(prompt_hash)
        if cached:
            return cached
            
        # Store response
        cache.set(prompt_hash, response)
    """
    
    def __init__(
        self, 
        ttl: int = 3600, 
        max_size: int = 1000,
        persist_path: Optional[str] = None,
        enabled: bool = True
    ):
        """
        Initialize the cache.
        
        Args:
            ttl: Time-to-live in seconds (default: 1 hour)
            max_size: Maximum number of cached entries
            persist_path: Optional path for disk persistence
            enabled: Whether caching is enabled
        """
        self.ttl = ttl
        self.max_size = max_size
        self.persist_path = Path(persist_path) if persist_path else None
        self.enabled = enabled
        
        # In-memory cache
        self._cache: TTLCache = TTLCache(maxsize=max_size, ttl=ttl)
        
        # Statistics
        self._hits = 0
        self._misses = 0
        self._total_saved_tokens = 0
        
        # Load persistent cache if available
        if self.persist_path and self.persist_path.exists():
            self._load_from_disk()
    
    @staticmethod
    def generate_key(prompt: str, model: str, **kwargs) -> str:
        """
        Generate a unique cache key for a request.
        
        Args:
            prompt: The prompt text
            model: The model name
            **kwargs: Additional parameters that affect the response
            
        Returns:
            A unique hash key
        """
        key_data = {
            "prompt": prompt,
            "model": model,
            **kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve a cached response.
        
        Args:
            key: The cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if not self.enabled:
            return None
            
        try:
            value = self._cache.get(key)
            if value is not None:
                self._hits += 1
                return value
            self._misses += 1
            return None
        except KeyError:
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any, estimated_tokens: int = 0) -> None:
        """
        Store a response in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            estimated_tokens: Estimated tokens saved by caching
        """
        if not self.enabled:
            return
            
        self._cache[key] = value
        self._total_saved_tokens += estimated_tokens
        
        # Persist to disk periodically
        if self.persist_path and len(self._cache) % 10 == 0:
            self._save_to_disk()
    
    def invalidate(self, key: str) -> bool:
        """
        Remove a specific entry from cache.
        
        Args:
            key: The cache key to invalidate
            
        Returns:
            True if entry was found and removed
        """
        try:
            del self._cache[key]
            return True
        except KeyError:
            return False
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        if self.persist_path and self.persist_path.exists():
            self.persist_path.unlink()
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.2f}%",
            "total_requests": total_requests,
            "current_size": len(self._cache),
            "max_size": self.max_size,
            "estimated_tokens_saved": self._total_saved_tokens,
            "estimated_cost_saved": self._total_saved_tokens * 0.00003  # Rough estimate
        }
    
    def _save_to_disk(self) -> None:
        """Persist cache to disk."""
        if not self.persist_path:
            return
            
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert TTLCache to regular dict for serialization
        cache_data = {
            "entries": dict(self._cache),
            "stats": {
                "hits": self._hits,
                "misses": self._misses,
                "saved_tokens": self._total_saved_tokens
            }
        }
        
        with open(self.persist_path, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        if not self.persist_path or not self.persist_path.exists():
            return
            
        try:
            with open(self.persist_path, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Restore entries
            for key, value in cache_data.get("entries", {}).items():
                self._cache[key] = value
                
            # Restore stats
            stats = cache_data.get("stats", {})
            self._hits = stats.get("hits", 0)
            self._misses = stats.get("misses", 0)
            self._total_saved_tokens = stats.get("saved_tokens", 0)
            
        except Exception:
            # If loading fails, start fresh
            pass
    
    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if a key is in the cache."""
        return key in self._cache


# Global cache instance
_global_cache: Optional[ResponseCache] = None


def get_cache(
    ttl: int = 3600,
    max_size: int = 1000,
    persist_path: Optional[str] = None
) -> ResponseCache:
    """
    Get or create the global cache instance.
    
    Args:
        ttl: Time-to-live in seconds
        max_size: Maximum cache size
        persist_path: Optional path for persistence
        
    Returns:
        ResponseCache instance
    """
    global _global_cache
    
    if _global_cache is None:
        enabled = os.getenv("CACHE_ENABLED", "true").lower() == "true"
        ttl = int(os.getenv("CACHE_TTL", str(ttl)))
        max_size = int(os.getenv("CACHE_MAX_SIZE", str(max_size)))
        
        _global_cache = ResponseCache(
            ttl=ttl,
            max_size=max_size,
            persist_path=persist_path,
            enabled=enabled
        )
    
    return _global_cache
