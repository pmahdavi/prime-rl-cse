#!/usr/bin/env python3
"""
Test script for the Modal MCP server.

This script verifies that the MCP server can be imported and tools are correctly defined.
For actual testing with an MCP client, use the server directly.
"""

import sys
import asyncio
from pathlib import Path

# Add the modal directory to the path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp_server import server, list_tools
    print("✅ Successfully imported MCP server")
except ImportError as e:
    print(f"❌ Failed to import MCP server: {e}")
    print("\nMake sure you have installed the MCP SDK:")
    print("  pip install mcp")
    sys.exit(1)


async def test_list_tools():
    """Test that tools are properly defined."""
    print("\n" + "="*60)
    print("Testing tool definitions...")
    print("="*60)
    
    tools = await list_tools()
    
    print(f"\n✅ Found {len(tools)} tools:")
    for tool in tools:
        print(f"\n  📦 {tool.name}")
        print(f"     {tool.description}")
        
        # Check required parameters
        if "required" in tool.inputSchema:
            required = tool.inputSchema["required"]
            if required:
                print(f"     Required: {', '.join(required)}")
        
        # Check all parameters
        if "properties" in tool.inputSchema:
            props = tool.inputSchema["properties"]
            print(f"     Parameters: {len(props)}")
            for param_name, param_def in props.items():
                param_type = param_def.get("type", "unknown")
                default = param_def.get("default")
                default_str = f" (default: {default})" if default is not None else ""
                print(f"       - {param_name}: {param_type}{default_str}")


async def test_modal_cli():
    """Test that Modal CLI is accessible."""
    print("\n" + "="*60)
    print("Testing Modal CLI availability...")
    print("="*60)
    
    import subprocess
    
    try:
        result = subprocess.run(
            ["modal", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        
        if result.returncode == 0:
            print(f"✅ Modal CLI is installed: {result.stdout.strip()}")
            
            # Check if authenticated
            result = subprocess.run(
                ["modal", "app", "list"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            if result.returncode == 0:
                print("✅ Modal is authenticated and working")
            else:
                print("⚠️  Modal CLI available but may not be authenticated")
                print("   Run: modal setup")
        else:
            print("❌ Modal CLI returned an error")
            
    except FileNotFoundError:
        print("❌ Modal CLI not found in PATH")
        print("   Install with: pip install modal")
    except subprocess.TimeoutExpired:
        print("⚠️  Modal CLI command timed out")


async def main():
    """Run all tests."""
    print("="*60)
    print("Modal MCP Server Test Suite")
    print("="*60)
    
    await test_list_tools()
    await test_modal_cli()
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
    print("\nTo use the MCP server:")
    print("1. Configure it in your Cursor settings (see MCP_README.md)")
    print("2. Restart Cursor")
    print("3. Use natural language to interact with Modal")
    print("\nExample: 'Deploy a training job with 8 GPUs'")


if __name__ == "__main__":
    asyncio.run(main())
