"""
MedGemma Agent Framework - Tool Registry

This module provides tool registration, discovery, and lifecycle management:
- Register and unregister tools
- Discover tools by name, category, or tags
- Manage tool initialization and cleanup
- Generate tool documentation

Usage:
    from core.registry import get_registry

    # Get the global registry
    registry = get_registry()

    # Register a tool class
    registry.register(MyToolClass)

    # Get a tool instance
    tool = await registry.get_tool("my_tool")

    # Execute a tool
    result = await registry.execute("my_tool", {"input": "data"})

    # List all tools
    tools = registry.list_tools()

    # Find tools by category
    imaging_tools = registry.find_by_category(ToolCategory.IMAGING)
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, Union

from core.config import get_config, MedicAItionConfig
from tools.base import (
    BaseTool,
    ToolCategory,
    ToolInput,
    ToolMetadata,
    ToolOutput,
)


logger = logging.getLogger(__name__)


class ToolRegistryError(Exception):
    """Exception raised for registry errors."""
    pass


class ToolNotFoundError(ToolRegistryError):
    """Exception raised when a tool is not found."""
    pass


class ToolAlreadyRegisteredError(ToolRegistryError):
    """Exception raised when attempting to register a duplicate tool."""
    pass


class ToolRegistry:
    """
    Central registry for all tools in the MedGemma Agent Framework.

    Manages tool registration, discovery, initialization, and lifecycle.
    Thread-safe and supports async operations.
    """

    def __init__(self, config: Optional[MedicAItionConfig] = None):
        """
        Initialize the tool registry.

        Args:
            config: Optional configuration. Uses global config if not provided.
        """
        self._config = config or get_config()
        self._tool_classes: Dict[str, Type[BaseTool]] = {}
        self._tool_instances: Dict[str, BaseTool] = {}
        self._initialized_tools: Set[str] = set()
        self._lock = asyncio.Lock()
        self._shutdown = False

    @property
    def config(self) -> MedicAItionConfig:
        """Get the configuration."""
        return self._config

    def register(
        self,
        tool_class: Type[BaseTool],
        override: bool = False
    ) -> None:
        """
        Register a tool class with the registry.

        Args:
            tool_class: The tool class to register
            override: If True, override existing tool with same name

        Raises:
            ToolAlreadyRegisteredError: If tool name already registered and override=False
            ValueError: If tool_class is not a valid BaseTool subclass
        """
        if not isinstance(tool_class, type) or not issubclass(tool_class, BaseTool):
            raise ValueError(f"{tool_class} is not a valid BaseTool subclass")

        name = tool_class.name

        if name in self._tool_classes and not override:
            raise ToolAlreadyRegisteredError(
                f"Tool '{name}' is already registered. Use override=True to replace."
            )

        # Check if tool is enabled in config
        if not self._config.is_tool_enabled(name):
            logger.info(f"Tool '{name}' is disabled in configuration, skipping registration")
            return

        self._tool_classes[name] = tool_class
        logger.info(f"Registered tool: {name} (v{tool_class.version})")

    def unregister(self, name: str) -> None:
        """
        Unregister a tool.

        Args:
            name: Name of the tool to unregister

        Raises:
            ToolNotFoundError: If tool is not registered
        """
        if name not in self._tool_classes:
            raise ToolNotFoundError(f"Tool '{name}' is not registered")

        # Clean up instance if exists
        if name in self._tool_instances:
            del self._tool_instances[name]
        self._initialized_tools.discard(name)
        del self._tool_classes[name]
        logger.info(f"Unregistered tool: {name}")

    def is_registered(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tool_classes

    async def get_tool(self, name: str) -> BaseTool:
        """
        Get an initialized tool instance.

        Args:
            name: Name of the tool

        Returns:
            Initialized tool instance

        Raises:
            ToolNotFoundError: If tool is not registered
        """
        if name not in self._tool_classes:
            raise ToolNotFoundError(f"Tool '{name}' is not registered")

        async with self._lock:
            # Return existing instance if available
            if name in self._tool_instances:
                return self._tool_instances[name]

            # Create and initialize new instance
            tool_class = self._tool_classes[name]
            instance = tool_class()

            # Initialize if not already done
            if name not in self._initialized_tools:
                await instance.setup()
                self._initialized_tools.add(name)

            self._tool_instances[name] = instance
            return instance

    async def execute(
        self,
        name: str,
        input_data: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> ToolOutput:
        """
        Execute a tool by name.

        Args:
            name: Name of the tool
            input_data: Input data for the tool
            timeout: Optional execution timeout in seconds

        Returns:
            Tool output

        Raises:
            ToolNotFoundError: If tool is not registered
        """
        tool = await self.get_tool(name)

        # Use config timeout if not specified
        if timeout is None:
            timeout = self._config.resources.tool_timeout_seconds

        return await tool.run(input_data, timeout=timeout)

    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tool_classes.keys())

    def list_metadata(self) -> List[ToolMetadata]:
        """List metadata for all registered tools."""
        metadata = []
        for name, tool_class in self._tool_classes.items():
            instance = tool_class()
            metadata.append(instance.get_metadata())
        return metadata

    def find_by_category(self, category: ToolCategory) -> List[str]:
        """
        Find tools by category.

        Args:
            category: Tool category to filter by

        Returns:
            List of tool names in the category
        """
        return [
            name for name, tool_class in self._tool_classes.items()
            if tool_class.category == category
        ]

    def find_by_tag(self, tag: str) -> List[str]:
        """
        Find tools by tag.

        Args:
            tag: Tag to search for

        Returns:
            List of tool names with the tag
        """
        matching = []
        for name, tool_class in self._tool_classes.items():
            instance = tool_class()
            metadata = instance.get_metadata()
            if tag in metadata.tags:
                matching.append(name)
        return matching

    def get_tool_info(self, name: str) -> Dict[str, Any]:
        """
        Get detailed information about a tool.

        Args:
            name: Name of the tool

        Returns:
            Dictionary with tool information

        Raises:
            ToolNotFoundError: If tool is not registered
        """
        if name not in self._tool_classes:
            raise ToolNotFoundError(f"Tool '{name}' is not registered")

        tool_class = self._tool_classes[name]
        instance = tool_class()
        metadata = instance.get_metadata()

        return {
            "name": name,
            "description": tool_class.description,
            "version": tool_class.version,
            "category": tool_class.category.value,
            "metadata": metadata.model_dump(),
            "input_schema": instance.get_input_schema(),
            "output_schema": instance.get_output_schema(),
            "function_spec": instance.to_function_spec(),
        }

    def get_all_function_specs(self) -> List[Dict[str, Any]]:
        """
        Get function specifications for all tools.

        Useful for LLM function calling integration.

        Returns:
            List of function specifications
        """
        specs = []
        for name, tool_class in self._tool_classes.items():
            instance = tool_class()
            specs.append(instance.to_function_spec())
        return specs

    async def initialize_all(self) -> None:
        """Initialize all registered tools."""
        for name in self._tool_classes:
            if name not in self._initialized_tools:
                await self.get_tool(name)
        logger.info(f"Initialized {len(self._initialized_tools)} tools")

    async def shutdown(self) -> None:
        """Shutdown and cleanup all tools."""
        self._shutdown = True
        for name, instance in self._tool_instances.items():
            try:
                await instance.teardown()
                logger.debug(f"Shut down tool: {name}")
            except Exception as e:
                logger.error(f"Error shutting down tool {name}: {e}")

        self._tool_instances.clear()
        self._initialized_tools.clear()
        logger.info("Tool registry shutdown complete")

    def auto_discover(
        self,
        package_path: str = "tools",
        exclude_patterns: Optional[List[str]] = None
    ) -> int:
        """
        Auto-discover and register tools from a package.

        Scans the specified package for tool classes and registers them.

        Args:
            package_path: Python package path to scan
            exclude_patterns: Patterns to exclude from discovery

        Returns:
            Number of tools discovered and registered
        """
        exclude_patterns = exclude_patterns or ["__", "base"]
        discovered = 0

        try:
            # Get the package directory
            package = importlib.import_module(package_path)
            if not hasattr(package, "__path__"):
                logger.warning(f"Package {package_path} has no __path__")
                return 0

            package_dir = Path(package.__path__[0])

            # Find all Python files recursively
            for py_file in package_dir.rglob("*.py"):
                # Skip excluded patterns
                if any(pattern in py_file.name for pattern in exclude_patterns):
                    continue

                # Build module path
                relative_path = py_file.relative_to(package_dir.parent)
                module_path = str(relative_path.with_suffix("")).replace("/", ".")

                try:
                    module = importlib.import_module(module_path)

                    # Find all BaseTool subclasses in module
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type) and
                            issubclass(attr, BaseTool) and
                            attr is not BaseTool and
                            hasattr(attr, "name") and
                            attr.name != "base_tool"
                        ):
                            try:
                                self.register(attr)
                                discovered += 1
                            except ToolAlreadyRegisteredError:
                                pass  # Skip already registered tools
                            except Exception as e:
                                logger.error(f"Error registering {attr_name}: {e}")

                except ImportError as e:
                    logger.debug(f"Could not import {module_path}: {e}")
                except Exception as e:
                    logger.error(f"Error processing {module_path}: {e}")

        except ImportError as e:
            logger.error(f"Could not import package {package_path}: {e}")

        logger.info(f"Auto-discovered {discovered} tools from {package_path}")
        return discovered

    def generate_docs(self, format: str = "markdown") -> str:
        """
        Generate documentation for all registered tools.

        Args:
            format: Output format ("markdown" or "json")

        Returns:
            Documentation string
        """
        if format == "json":
            import json
            docs = {
                "tools": [
                    self.get_tool_info(name)
                    for name in sorted(self._tool_classes.keys())
                ]
            }
            return json.dumps(docs, indent=2)

        # Markdown format
        lines = ["# MedGemma Agent Tools\n"]

        # Group by category
        by_category: Dict[ToolCategory, List[str]] = {}
        for name, tool_class in self._tool_classes.items():
            category = tool_class.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(name)

        for category in ToolCategory:
            if category not in by_category:
                continue

            lines.append(f"\n## {category.value.title()}\n")

            for name in sorted(by_category[category]):
                tool_class = self._tool_classes[name]
                instance = tool_class()
                metadata = instance.get_metadata()

                lines.append(f"### {name}\n")
                lines.append(f"**Version:** {tool_class.version}\n")
                lines.append(f"**Description:** {tool_class.description}\n")

                if metadata.tags:
                    lines.append(f"**Tags:** {', '.join(metadata.tags)}\n")

                lines.append("\n**Input Schema:**\n```json\n")
                import json
                lines.append(json.dumps(instance.get_input_schema(), indent=2))
                lines.append("\n```\n")

        return "".join(lines)


# Global registry instance
_registry: Optional[ToolRegistry] = None


def get_registry(config: Optional[MedicAItionConfig] = None) -> ToolRegistry:
    """
    Get the global tool registry instance.

    Args:
        config: Optional configuration for first initialization

    Returns:
        Global ToolRegistry instance
    """
    global _registry
    if _registry is None:
        _registry = ToolRegistry(config)
    return _registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _registry
    _registry = None


async def register_all_tools() -> ToolRegistry:
    """
    Register all built-in tools.

    This function imports and registers all tools from the tools package.

    Returns:
        The registry with all tools registered
    """
    registry = get_registry()

    # Auto-discover tools from the tools package
    registry.auto_discover("tools")

    return registry
