"""
Rate Limiting Module

Provides rate limiting functionality for API calls to prevent exceeding
provider limits and ensure reliable operation.

Author: ML Engineer
Date: 2025-01-06
"""

import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 60
    tokens_per_minute: int = 90000
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""
    def __init__(self, wait_time: float, limit_type: str = "requests"):
        self.wait_time = wait_time
        self.limit_type = limit_type
        super().__init__(
            f"Rate limit exceeded ({limit_type}). Wait {wait_time:.2f} seconds."
        )


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    
    Features:
    - Request per minute limiting
    - Token per minute limiting
    - Thread-safe operation
    - Automatic waiting/blocking
    - Statistics tracking
    
    Usage:
        limiter = RateLimiter(requests_per_minute=60, tokens_per_minute=90000)
        
        # Before making a request
        limiter.acquire(estimated_tokens=1000)
        
        # After request completes
        limiter.record_usage(actual_tokens=950)
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        tokens_per_minute: int = 90000,
        max_retries: int = 3
    ):
        """
        Initialize the rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute
            tokens_per_minute: Maximum tokens per minute
            max_retries: Maximum retry attempts on rate limit
        """
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self.max_retries = max_retries
        
        # Sliding window for requests
        self._request_times: deque = deque()
        self._token_usage: deque = deque()  # (timestamp, tokens)
        
        # Thread safety
        self._lock = Lock()
        
        # Statistics
        self._total_requests = 0
        self._total_tokens = 0
        self._rate_limit_hits = 0
        self._total_wait_time = 0.0
    
    def acquire(self, estimated_tokens: int = 0, block: bool = True) -> bool:
        """
        Acquire permission to make a request.
        
        Args:
            estimated_tokens: Estimated tokens for this request
            block: If True, wait until rate limit allows; if False, raise exception
            
        Returns:
            True if acquired successfully
            
        Raises:
            RateLimitExceeded: If block=False and rate limit would be exceeded
        """
        with self._lock:
            now = time.time()
            window_start = now - 60  # 1-minute sliding window
            
            # Clean up old entries
            self._cleanup_old_entries(window_start)
            
            # Check request limit
            if len(self._request_times) >= self.rpm:
                wait_time = self._request_times[0] - window_start
                if block:
                    self._rate_limit_hits += 1
                    self._total_wait_time += wait_time
                    time.sleep(wait_time + 0.1)
                    return self.acquire(estimated_tokens, block)
                else:
                    raise RateLimitExceeded(wait_time, "requests")
            
            # Check token limit
            current_tokens = sum(t[1] for t in self._token_usage)
            if current_tokens + estimated_tokens > self.tpm:
                if self._token_usage:
                    wait_time = self._token_usage[0][0] - window_start
                    if block:
                        self._rate_limit_hits += 1
                        self._total_wait_time += wait_time
                        time.sleep(wait_time + 0.1)
                        return self.acquire(estimated_tokens, block)
                    else:
                        raise RateLimitExceeded(wait_time, "tokens")
            
            # Record this request
            self._request_times.append(now)
            if estimated_tokens > 0:
                self._token_usage.append((now, estimated_tokens))
            
            self._total_requests += 1
            
            return True
    
    def record_usage(self, actual_tokens: int) -> None:
        """
        Record actual token usage after a request completes.
        
        Args:
            actual_tokens: Actual tokens used
        """
        with self._lock:
            self._total_tokens += actual_tokens
            
            # Update the most recent token entry with actual value
            if self._token_usage:
                now, estimated = self._token_usage[-1]
                self._token_usage[-1] = (now, actual_tokens)
    
    def _cleanup_old_entries(self, window_start: float) -> None:
        """Remove entries older than the sliding window."""
        while self._request_times and self._request_times[0] < window_start:
            self._request_times.popleft()
        
        while self._token_usage and self._token_usage[0][0] < window_start:
            self._token_usage.popleft()
    
    def get_current_usage(self) -> dict:
        """
        Get current rate limit usage.
        
        Returns:
            Dictionary with current usage statistics
        """
        with self._lock:
            now = time.time()
            window_start = now - 60
            self._cleanup_old_entries(window_start)
            
            current_requests = len(self._request_times)
            current_tokens = sum(t[1] for t in self._token_usage)
            
            return {
                "requests_used": current_requests,
                "requests_limit": self.rpm,
                "requests_remaining": self.rpm - current_requests,
                "tokens_used": current_tokens,
                "tokens_limit": self.tpm,
                "tokens_remaining": self.tpm - current_tokens,
                "utilization": {
                    "requests": f"{(current_requests / self.rpm) * 100:.1f}%",
                    "tokens": f"{(current_tokens / self.tpm) * 100:.1f}%"
                }
            }
    
    def get_stats(self) -> dict:
        """
        Get rate limiter statistics.
        
        Returns:
            Dictionary with overall statistics
        """
        return {
            "total_requests": self._total_requests,
            "total_tokens": self._total_tokens,
            "rate_limit_hits": self._rate_limit_hits,
            "total_wait_time": f"{self._total_wait_time:.2f}s",
            "average_tokens_per_request": (
                self._total_tokens / self._total_requests 
                if self._total_requests > 0 else 0
            )
        }
    
    def reset_stats(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._total_requests = 0
            self._total_tokens = 0
            self._rate_limit_hits = 0
            self._total_wait_time = 0.0


def create_retry_decorator(
    max_retries: int = 3,
    min_wait: float = 1,
    max_wait: float = 60
):
    """
    Create a retry decorator with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        min_wait: Minimum wait time between retries
        max_wait: Maximum wait time between retries
        
    Returns:
        Configured retry decorator
    """
    return retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_if_exception_type((RateLimitExceeded, ConnectionError, TimeoutError)),
        reraise=True
    )


# Global rate limiter instance
_global_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """
    Get or create the global rate limiter instance.
    
    Returns:
        RateLimiter instance
    """
    global _global_limiter
    
    if _global_limiter is None:
        rpm = int(os.getenv("RATE_LIMIT_RPM", "60"))
        tpm = int(os.getenv("RATE_LIMIT_TPM", "90000"))
        
        _global_limiter = RateLimiter(
            requests_per_minute=rpm,
            tokens_per_minute=tpm
        )
    
    return _global_limiter
