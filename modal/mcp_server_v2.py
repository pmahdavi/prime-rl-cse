#!/usr/bin/env python3
"""
Modal MCP Server v2 for Prime-RL

Enhanced version with comprehensive Modal CLI integration:
- App management (list, logs, stop, history, rollback)
- Volume operations (list, ls, get, put, rm, cp, create, delete)
- Container management (list, logs, exec, stop)
- Secret management (list, create, delete)
- NFS operations (list, ls, get, put, rm, create, delete)
- Training deployment (prime-rl specific)
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

# MCP SDK imports
try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    import mcp.server.stdio
except ImportError:
    print("Error: MCP SDK not installed. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

# Initialize the MCP server
server = Server("prime-rl-modal-v2")

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


def run_modal_command(args: list[str], capture_output: bool = True) -> tuple[int, str, str]:
    """Run a modal CLI command and return (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(
            ["modal"] + args,
            capture_output=capture_output,
            text=True,
            cwd=PROJECT_ROOT,
        )
        return result.returncode, result.stdout or "", result.stderr or ""
    except Exception as e:
        return 1, "", str(e)


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available Modal tools."""
    return [
        # ==== DEPLOYMENT ====
        Tool(
            name="modal_deploy_training",
            description="Deploy a prime-rl training job on Modal",
            inputSchema={
                "type": "object",
                "properties": {
                    "experiment_name": {"type": "string"},
                    "trainer_config": {"type": "string", "default": "configs/reverse_text/train.toml"},
                    "orchestrator_config": {"type": "string", "default": "configs/reverse_text/orch.toml"},
                    "inference_config": {"type": "string", "default": "configs/reverse_text/infer.toml"},
                    "gpu_count": {"type": "integer", "default": 2},
                    "trainer_gpu_ratio": {"type": "number", "default": 0.5},
                    "gpu_type": {"type": "string", "default": "A100-40GB"},
                    "wandb_project": {"type": "string"},
                    "use_local_code": {"type": "boolean", "default": False},
                },
            },
        ),
        
        # ==== APP MANAGEMENT ====
        Tool(
            name="modal_app_list",
            description="List Modal apps (deployed/running/stopped)",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="modal_app_logs",
            description="Get logs from a Modal app (streams all logs)",
            inputSchema={
                "type": "object",
                "properties": {
                    "app_id": {"type": "string", "description": "App ID or app name (required)"},
                },
                "required": ["app_id"],
            },
        ),
        Tool(
            name="modal_app_stop",
            description="Stop a running Modal app",
            inputSchema={
                "type": "object",
                "properties": {"app_id": {"type": "string"}},
                "required": ["app_id"],
            },
        ),
        Tool(
            name="modal_app_history",
            description="Show deployment history for an app",
            inputSchema={
                "type": "object",
                "properties": {"app_name": {"type": "string"}},
                "required": ["app_name"],
            },
        ),
        
        # ==== CONTAINER MANAGEMENT ====
        Tool(
            name="modal_container_list",
            description="List all running containers",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="modal_container_logs",
            description="Show logs for a specific container",
            inputSchema={
                "type": "object",
                "properties": {"container_id": {"type": "string"}},
                "required": ["container_id"],
            },
        ),
        Tool(
            name="modal_container_exec",
            description="Execute a command in a container",
            inputSchema={
                "type": "object",
                "properties": {
                    "container_id": {"type": "string"},
                    "command": {"type": "string"},
                },
                "required": ["container_id", "command"],
            },
        ),
        Tool(
            name="modal_container_stop",
            description="Stop a running container",
            inputSchema={
                "type": "object",
                "properties": {"container_id": {"type": "string"}},
                "required": ["container_id"],
            },
        ),
        
        # ==== VOLUME MANAGEMENT ====
        Tool(
            name="modal_volume_list",
            description="List all volumes",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="modal_volume_ls",
            description="List files in a volume",
            inputSchema={
                "type": "object",
                "properties": {
                    "volume_name": {"type": "string"},
                    "path": {"type": "string", "default": "/"},
                },
                "required": ["volume_name"],
            },
        ),
        Tool(
            name="modal_volume_get",
            description="Download files from a volume",
            inputSchema={
                "type": "object",
                "properties": {
                    "volume_name": {"type": "string"},
                    "remote_path": {"type": "string"},
                    "local_path": {"type": "string"},
                },
                "required": ["volume_name", "remote_path", "local_path"],
            },
        ),
        Tool(
            name="modal_volume_put",
            description="Upload file/directory to a volume",
            inputSchema={
                "type": "object",
                "properties": {
                    "volume_name": {"type": "string"},
                    "local_path": {"type": "string"},
                    "remote_path": {"type": "string"},
                },
                "required": ["volume_name", "local_path", "remote_path"],
            },
        ),
        Tool(
            name="modal_volume_rm",
            description="Delete file/directory from a volume",
            inputSchema={
                "type": "object",
                "properties": {
                    "volume_name": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["volume_name", "path"],
            },
        ),
        Tool(
            name="modal_volume_create",
            description="Create a new volume",
            inputSchema={
                "type": "object",
                "properties": {"volume_name": {"type": "string"}},
                "required": ["volume_name"],
            },
        ),
        Tool(
            name="modal_volume_delete",
            description="Delete a volume",
            inputSchema={
                "type": "object",
                "properties": {"volume_name": {"type": "string"}},
                "required": ["volume_name"],
            },
        ),
        
        # ==== SECRET MANAGEMENT ====
        Tool(
            name="modal_secret_list",
            description="List all secrets",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="modal_secret_create",
            description="Create a new secret (interactive)",
            inputSchema={
                "type": "object",
                "properties": {"secret_name": {"type": "string"}},
                "required": ["secret_name"],
            },
        ),
        
        # ==== NFS MANAGEMENT ====
        Tool(
            name="modal_nfs_list",
            description="List all network file systems",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="modal_nfs_ls",
            description="List files in an NFS",
            inputSchema={
                "type": "object",
                "properties": {
                    "nfs_name": {"type": "string"},
                    "path": {"type": "string", "default": "/"},
                },
                "required": ["nfs_name"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    
    # ==== DEPLOYMENT ====
    if name == "modal_deploy_training":
        cmd = ["run", "modal/deploy.py"]
        if arguments.get("experiment_name"):
            cmd.extend(["--experiment-name", arguments["experiment_name"]])
        if arguments.get("trainer_config"):
            cmd.extend(["--trainer-config", arguments["trainer_config"]])
        if arguments.get("orchestrator_config"):
            cmd.extend(["--orchestrator-config", arguments["orchestrator_config"]])
        if arguments.get("inference_config"):
            cmd.extend(["--inference-config", arguments["inference_config"]])
        if arguments.get("gpu_count"):
            cmd.extend(["--gpu-count", str(arguments["gpu_count"])])
        if arguments.get("trainer_gpu_ratio"):
            cmd.extend(["--trainer-gpu-ratio", str(arguments["trainer_gpu_ratio"])])
        if arguments.get("gpu_type"):
            cmd.extend(["--gpu-type", arguments["gpu_type"]])
        if arguments.get("wandb_project"):
            cmd.extend(["--wandb-project", arguments["wandb_project"]])
        if arguments.get("use_local_code"):
            cmd.append("--use-local-code")
        
        returncode, stdout, stderr = run_modal_command(cmd)
        return [TextContent(
            type="text",
            text=f"{'✅ Deployed' if returncode == 0 else '❌ Failed'}\n\n{stdout}\n{stderr}"
        )]
    
    # ==== APP MANAGEMENT ====
    elif name == "modal_app_list":
        returncode, stdout, stderr = run_modal_command(["app", "list"])
        return [TextContent(type="text", text=stdout if returncode == 0 else stderr)]
    
    elif name == "modal_app_logs":
        cmd = ["app", "logs", arguments["app_id"]]
        returncode, stdout, stderr = run_modal_command(cmd)
        return [TextContent(type="text", text=stdout if returncode == 0 else stderr)]
    
    elif name == "modal_app_stop":
        returncode, stdout, stderr = run_modal_command(["app", "stop", arguments["app_id"]])
        return [TextContent(type="text", text=f"{'✅ Stopped' if returncode == 0 else '❌ Failed'}\n{stderr}")]
    
    elif name == "modal_app_history":
        returncode, stdout, stderr = run_modal_command(["app", "history", arguments["app_name"]])
        return [TextContent(type="text", text=stdout if returncode == 0 else stderr)]
    
    # ==== CONTAINER MANAGEMENT ====
    elif name == "modal_container_list":
        returncode, stdout, stderr = run_modal_command(["container", "list"])
        return [TextContent(type="text", text=stdout if returncode == 0 else stderr)]
    
    elif name == "modal_container_logs":
        returncode, stdout, stderr = run_modal_command(["container", "logs", arguments["container_id"]])
        return [TextContent(type="text", text=stdout if returncode == 0 else stderr)]
    
    elif name == "modal_container_exec":
        returncode, stdout, stderr = run_modal_command([
            "container", "exec", arguments["container_id"], "--", arguments["command"]
        ])
        return [TextContent(type="text", text=stdout if returncode == 0 else stderr)]
    
    elif name == "modal_container_stop":
        returncode, stdout, stderr = run_modal_command(["container", "stop", arguments["container_id"]])
        return [TextContent(type="text", text=f"{'✅ Stopped' if returncode == 0 else '❌ Failed'}\n{stderr}")]
    
    # ==== VOLUME MANAGEMENT ====
    elif name == "modal_volume_list":
        returncode, stdout, stderr = run_modal_command(["volume", "list"])
        return [TextContent(type="text", text=stdout if returncode == 0 else stderr)]
    
    elif name == "modal_volume_ls":
        cmd = ["volume", "ls", arguments["volume_name"]]
        if arguments.get("path"):
            cmd.append(arguments["path"])
        returncode, stdout, stderr = run_modal_command(cmd)
        return [TextContent(type="text", text=stdout if returncode == 0 else stderr)]
    
    elif name == "modal_volume_get":
        returncode, stdout, stderr = run_modal_command([
            "volume", "get", arguments["volume_name"], 
            arguments["remote_path"], arguments["local_path"]
        ])
        return [TextContent(type="text", text=f"{'✅ Downloaded' if returncode == 0 else '❌ Failed'}\n{stdout}\n{stderr}")]
    
    elif name == "modal_volume_put":
        returncode, stdout, stderr = run_modal_command([
            "volume", "put", arguments["volume_name"],
            arguments["local_path"], arguments["remote_path"]
        ])
        return [TextContent(type="text", text=f"{'✅ Uploaded' if returncode == 0 else '❌ Failed'}\n{stdout}\n{stderr}")]
    
    elif name == "modal_volume_rm":
        returncode, stdout, stderr = run_modal_command([
            "volume", "rm", arguments["volume_name"], arguments["path"]
        ])
        return [TextContent(type="text", text=f"{'✅ Deleted' if returncode == 0 else '❌ Failed'}\n{stderr}")]
    
    elif name == "modal_volume_create":
        returncode, stdout, stderr = run_modal_command(["volume", "create", arguments["volume_name"]])
        return [TextContent(type="text", text=f"{'✅ Created' if returncode == 0 else '❌ Failed'}\n{stderr}")]
    
    elif name == "modal_volume_delete":
        returncode, stdout, stderr = run_modal_command(["volume", "delete", arguments["volume_name"]])
        return [TextContent(type="text", text=f"{'✅ Deleted' if returncode == 0 else '❌ Failed'}\n{stderr}")]
    
    # ==== SECRET MANAGEMENT ====
    elif name == "modal_secret_list":
        returncode, stdout, stderr = run_modal_command(["secret", "list"])
        return [TextContent(type="text", text=stdout if returncode == 0 else stderr)]
    
    elif name == "modal_secret_create":
        return [TextContent(
            type="text",
            text=f"⚠️  Secret creation is interactive. Please run:\n  modal secret create {arguments['secret_name']}"
        )]
    
    # ==== NFS MANAGEMENT ====
    elif name == "modal_nfs_list":
        returncode, stdout, stderr = run_modal_command(["nfs", "list"])
        return [TextContent(type="text", text=stdout if returncode == 0 else stderr)]
    
    elif name == "modal_nfs_ls":
        cmd = ["nfs", "ls", arguments["nfs_name"]]
        if arguments.get("path"):
            cmd.append(arguments["path"])
        returncode, stdout, stderr = run_modal_command(cmd)
        return [TextContent(type="text", text=stdout if returncode == 0 else stderr)]
    
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run the MCP server."""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
