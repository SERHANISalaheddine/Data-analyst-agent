"""
=============================================================================
MCP - Model Context Protocol Client Package
=============================================================================

This package contains the MCP client for connecting to external services
via the Model Context Protocol through Smithery.

The main export is get_sheets_tools() which loads all Google Sheets MCP
tools as LangChain-compatible tools that can be bound to an LLM.
"""

from .sheets_client import get_sheets_tools, get_sheets_tools_async

__all__ = ["get_sheets_tools", "get_sheets_tools_async"]

# NOTE: This package is named 'sheets_mcp' (not 'mcp') to avoid shadowing
# the installed 'mcp' package from pip which langchain_mcp_adapters depends on.
