"""
LLM Client Module

Provides a unified interface for interacting with various LLM providers
(OpenAI, Anthropic, Ollama) with built-in caching, rate limiting, and
error handling.

Author: ML Engineer
Date: 2025-01-06
"""

import json
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from ..utils.cache import ResponseCache, get_cache
from ..utils.rate_limiter import RateLimiter, get_rate_limiter, create_retry_decorator
from .prompts import PromptBuilder

# Load environment variables
load_dotenv()


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class TestSuggestion(BaseModel):
    """Represents a test suggestion from the LLM."""
    target_bin: str = Field(..., description="Full path to the target bin")
    priority: str = Field(..., description="Priority level: high/medium/low")
    difficulty: str = Field(..., description="Difficulty level: easy/medium/hard")
    suggestion: str = Field(..., description="Detailed test scenario description")
    test_outline: list[str] = Field(default_factory=list, description="Step-by-step test outline")
    dependencies: list[str] = Field(default_factory=list, description="Prerequisites and dependencies")
    reasoning: str = Field(..., description="Explanation of why this approach will work")
    
    # Computed fields for prioritization
    priority_score: Optional[float] = Field(None, description="Computed priority score")


class LLMResponse(BaseModel):
    """Response from an LLM query."""
    suggestions: list[TestSuggestion] = Field(default_factory=list)
    raw_response: Optional[str] = Field(None, description="Raw LLM response")
    tokens_used: int = Field(0, description="Total tokens used")
    cached: bool = Field(False, description="Whether response was from cache")
    model: str = Field("", description="Model used for generation")


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate the number of tokens in a text."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self._last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using OpenAI."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=4096,
                response_format={"type": "json_object"}  # Ensure JSON output
            )
            
            self._last_usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using a simple heuristic (4 chars per token)."""
        return len(text) // 4
    
    @property
    def last_usage(self) -> dict:
        """Get token usage from last request."""
        return self._last_usage


class AnthropicClient(BaseLLMClient):
    """Anthropic (Claude) API client."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """Initialize the Anthropic client."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")
        
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
        
        if not self.api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable.")
        
        self.client = Anthropic(api_key=self.api_key)
        self._last_usage = {"input_tokens": 0, "output_tokens": 0}
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using Anthropic Claude."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=system_prompt or "You are a helpful assistant.",
                messages=[
                    {"role": "user", "content": prompt + "\n\nRespond with valid JSON only."}
                ]
            )
            
            self._last_usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }
            
            return response.content[0].text
            
        except Exception as e:
            raise RuntimeError(f"Anthropic API error: {str(e)}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens using a simple heuristic."""
        return len(text) // 4
    
    @property
    def last_usage(self) -> dict:
        """Get token usage from last request."""
        return self._last_usage


class OllamaClient(BaseLLMClient):
    """Ollama (local LLM) client."""
    
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None):
        """Initialize the Ollama client."""
        import urllib.request
        import urllib.error
        
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama2")
        self._last_tokens = 0
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using Ollama."""
        import urllib.request
        import urllib.error
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        
        data = json.dumps({
            "model": self.model,
            "prompt": full_prompt + "\n\nRespond with valid JSON only.",
            "stream": False,
            "format": "json"
        }).encode('utf-8')
        
        try:
            req = urllib.request.Request(
                f"{self.host}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode('utf-8'))
                self._last_tokens = result.get("eval_count", 0)
                return result.get("response", "")
                
        except urllib.error.URLError as e:
            raise RuntimeError(f"Ollama connection error: {str(e)}. Is Ollama running?")
        except Exception as e:
            raise RuntimeError(f"Ollama error: {str(e)}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens."""
        return len(text) // 4
    
    @property
    def last_usage(self) -> dict:
        """Get token usage from last request."""
        return {"total_tokens": self._last_tokens}


class LLMClient:
    """
    Unified LLM client with caching, rate limiting, and error handling.
    
    This is the main interface for LLM interactions in the coverage analyzer.
    It provides:
    - Support for multiple LLM providers (OpenAI, Anthropic, Ollama)
    - Response caching to reduce API costs
    - Rate limiting to prevent quota exhaustion
    - Automatic retries with exponential backoff
    - Structured output parsing
    
    Usage:
        client = LLMClient(provider=LLMProvider.OPENAI)
        response = client.generate_suggestions(report, uncovered_bin)
    """
    
    def __init__(
        self,
        provider: LLMProvider = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache: Optional[ResponseCache] = None,
        rate_limiter: Optional[RateLimiter] = None
    ):
        """
        Initialize the LLM client.
        
        Args:
            provider: LLM provider to use
            api_key: API key (optional, will use env vars)
            model: Model name (optional, will use env vars)
            cache: Response cache instance
            rate_limiter: Rate limiter instance
        """
        # Determine provider from environment if not specified
        if provider is None:
            provider_str = os.getenv("LLM_PROVIDER", "openai").lower()
            provider = LLMProvider(provider_str)
        
        self.provider = provider
        self.cache = cache or get_cache()
        self.rate_limiter = rate_limiter or get_rate_limiter()
        self.prompt_builder = PromptBuilder()
        
        # Initialize the appropriate client
        if provider == LLMProvider.OPENAI:
            self._client = OpenAIClient(api_key=api_key, model=model)
        elif provider == LLMProvider.ANTHROPIC:
            self._client = AnthropicClient(api_key=api_key, model=model)
        elif provider == LLMProvider.OLLAMA:
            self._client = OllamaClient(model=model)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Retry decorator
        self._retry = create_retry_decorator(max_retries=3)
    
    def generate_suggestion(
        self,
        report: 'CoverageReport',
        uncovered_bin: 'UncoveredBinInfo'
    ) -> TestSuggestion:
        """
        Generate a test suggestion for a single uncovered bin.
        
        Args:
            report: The parsed coverage report
            uncovered_bin: The uncovered bin to target
            
        Returns:
            TestSuggestion object
        """
        from ..parser.coverage_parser import CoverageReport, UncoveredBinInfo
        
        # Build the prompt
        prompt = self.prompt_builder.build_suggestion_prompt(report, uncovered_bin)
        system_prompt = self.prompt_builder.get_system_prompt()
        
        # Check cache
        cache_key = self.cache.generate_key(
            prompt=prompt,
            model=getattr(self._client, 'model', 'unknown')
        )
        
        cached_response = self.cache.get(cache_key)
        if cached_response:
            return self._parse_suggestion(cached_response, cached=True)
        
        # Rate limit and generate
        estimated_tokens = self._client.estimate_tokens(prompt)
        self.rate_limiter.acquire(estimated_tokens)
        
        try:
            response = self._client.generate(prompt, system_prompt)
            
            # Record actual usage
            usage = getattr(self._client, 'last_usage', {})
            actual_tokens = usage.get('total_tokens', usage.get('input_tokens', 0) + usage.get('output_tokens', 0))
            self.rate_limiter.record_usage(actual_tokens)
            
            # Cache the response
            self.cache.set(cache_key, response, actual_tokens)
            
            return self._parse_suggestion(response, cached=False)
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate suggestion: {str(e)}")
    
    def generate_all_suggestions(
        self,
        report: 'CoverageReport',
        batch_mode: bool = True
    ) -> LLMResponse:
        """
        Generate suggestions for all uncovered bins.
        
        Args:
            report: The parsed coverage report
            batch_mode: If True, use a single prompt for all bins
            
        Returns:
            LLMResponse with all suggestions
        """
        from ..parser.coverage_parser import CoverageReport
        
        if batch_mode:
            return self._generate_batch_suggestions(report)
        else:
            return self._generate_individual_suggestions(report)
    
    def _generate_batch_suggestions(self, report: 'CoverageReport') -> LLMResponse:
        """Generate all suggestions in a single batch request."""
        prompt = self.prompt_builder.build_batch_prompt(report)
        system_prompt = self.prompt_builder.get_system_prompt()
        
        # Check cache
        cache_key = self.cache.generate_key(
            prompt=prompt,
            model=getattr(self._client, 'model', 'unknown')
        )
        
        cached_response = self.cache.get(cache_key)
        if cached_response:
            return self._parse_batch_response(cached_response, cached=True)
        
        # Rate limit and generate
        estimated_tokens = self._client.estimate_tokens(prompt)
        self.rate_limiter.acquire(estimated_tokens)
        
        response = self._client.generate(prompt, system_prompt)
        
        # Record usage and cache
        usage = getattr(self._client, 'last_usage', {})
        actual_tokens = usage.get('total_tokens', usage.get('input_tokens', 0) + usage.get('output_tokens', 0))
        self.rate_limiter.record_usage(actual_tokens)
        self.cache.set(cache_key, response, actual_tokens)
        
        return self._parse_batch_response(response, cached=False)
    
    def _generate_individual_suggestions(self, report: 'CoverageReport') -> LLMResponse:
        """Generate suggestions one by one for each uncovered bin."""
        suggestions = []
        total_tokens = 0
        
        # Process regular uncovered bins
        for uncovered_bin in report.uncovered_bins:
            try:
                suggestion = self.generate_suggestion(report, uncovered_bin)
                suggestions.append(suggestion)
                
                usage = getattr(self._client, 'last_usage', {})
                total_tokens += usage.get('total_tokens', 0)
                
            except Exception as e:
                print(f"Warning: Failed to generate suggestion for {uncovered_bin.full_path}: {e}")
        
        # Process cross-coverage
        for xc in report.cross_coverage:
            try:
                xc_suggestions = self._generate_cross_suggestions(report, xc)
                suggestions.extend(xc_suggestions)
            except Exception as e:
                print(f"Warning: Failed to generate cross-coverage suggestions for {xc.name}: {e}")
        
        return LLMResponse(
            suggestions=suggestions,
            tokens_used=total_tokens,
            cached=False,
            model=getattr(self._client, 'model', 'unknown')
        )
    
    def _generate_cross_suggestions(
        self,
        report: 'CoverageReport',
        cross_coverage: 'CrossCoverageInfo'
    ) -> list[TestSuggestion]:
        """Generate suggestions for cross-coverage holes."""
        from ..parser.coverage_parser import CrossCoverageInfo
        
        prompt = self.prompt_builder.build_cross_coverage_prompt(report, cross_coverage)
        system_prompt = self.prompt_builder.get_system_prompt()
        
        # Check cache
        cache_key = self.cache.generate_key(
            prompt=prompt,
            model=getattr(self._client, 'model', 'unknown')
        )
        
        cached_response = self.cache.get(cache_key)
        if cached_response:
            return self._parse_cross_response(cached_response)
        
        # Rate limit and generate
        estimated_tokens = self._client.estimate_tokens(prompt)
        self.rate_limiter.acquire(estimated_tokens)
        
        response = self._client.generate(prompt, system_prompt)
        
        # Record usage and cache
        usage = getattr(self._client, 'last_usage', {})
        actual_tokens = usage.get('total_tokens', 0)
        self.rate_limiter.record_usage(actual_tokens)
        self.cache.set(cache_key, response, actual_tokens)
        
        return self._parse_cross_response(response)
    
    def _parse_suggestion(self, response: str, cached: bool = False) -> TestSuggestion:
        """Parse a single suggestion from LLM response."""
        try:
            # Clean up the response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            data = json.loads(response)
            
            return TestSuggestion(
                target_bin=data.get("target_bin", "unknown"),
                priority=data.get("priority", "medium"),
                difficulty=data.get("difficulty", "medium"),
                suggestion=data.get("suggestion", ""),
                test_outline=data.get("test_outline", []),
                dependencies=data.get("dependencies", []),
                reasoning=data.get("reasoning", "")
            )
            
        except json.JSONDecodeError as e:
            # Try to extract useful information even from malformed JSON
            return TestSuggestion(
                target_bin="parse_error",
                priority="medium",
                difficulty="medium",
                suggestion=response[:500],
                test_outline=[],
                dependencies=[],
                reasoning=f"Failed to parse LLM response: {str(e)}"
            )
    
    def _parse_batch_response(self, response: str, cached: bool = False) -> LLMResponse:
        """Parse a batch response containing multiple suggestions."""
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            data = json.loads(response)
            suggestions = []
            
            for s in data.get("suggestions", []):
                suggestions.append(TestSuggestion(
                    target_bin=s.get("target_bin", "unknown"),
                    priority=s.get("priority", "medium"),
                    difficulty=s.get("difficulty", "medium"),
                    suggestion=s.get("suggestion", ""),
                    test_outline=s.get("test_outline", []),
                    dependencies=s.get("dependencies", []),
                    reasoning=s.get("reasoning", "")
                ))
            
            return LLMResponse(
                suggestions=suggestions,
                raw_response=response,
                cached=cached,
                model=getattr(self._client, 'model', 'unknown')
            )
            
        except json.JSONDecodeError:
            return LLMResponse(
                suggestions=[],
                raw_response=response,
                cached=cached,
                model=getattr(self._client, 'model', 'unknown')
            )
    
    def _parse_cross_response(self, response: str) -> list[TestSuggestion]:
        """Parse cross-coverage suggestions from LLM response."""
        try:
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            data = json.loads(response)
            suggestions = []
            
            for s in data.get("suggestions", []):
                suggestions.append(TestSuggestion(
                    target_bin=s.get("target_bin", "unknown"),
                    priority=s.get("priority", "medium"),
                    difficulty=s.get("difficulty", "medium"),
                    suggestion=s.get("suggestion", ""),
                    test_outline=s.get("test_outline", []),
                    dependencies=s.get("dependencies", []),
                    reasoning=s.get("reasoning", "")
                ))
            
            return suggestions
            
        except json.JSONDecodeError:
            return []
    
    def get_stats(self) -> dict:
        """Get combined statistics from cache and rate limiter."""
        return {
            "cache": self.cache.get_stats(),
            "rate_limiter": self.rate_limiter.get_stats(),
            "provider": self.provider.value,
            "model": getattr(self._client, 'model', 'unknown')
        }


def create_llm_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None
) -> LLMClient:
    """
    Factory function to create an LLM client.
    
    Args:
        provider: Provider name ("openai", "anthropic", "ollama")
        api_key: API key (optional)
        model: Model name (optional)
        
    Returns:
        Configured LLMClient instance
    """
    if provider:
        provider_enum = LLMProvider(provider.lower())
    else:
        provider_enum = None
    
    return LLMClient(provider=provider_enum, api_key=api_key, model=model)
