"""
Backward-compatibility re-export module. QueryNormalizer has been unified into query_analyzer.py.
"""
from query_analyzer import QueryNormalizer

__all__ = ["QueryNormalizer"]
