# Building Agentic Applications with Claude Agent SDK for Python

A comprehensive guide to leveraging the Claude Agent SDK for building sophisticated AI agent systems.

## Table of Contents

- [Overview](#overview)
- [Core Concepts](#core-concepts)
- [Getting Started](#getting-started)
- [Building Custom Tools](#building-custom-tools)
- [Control Flow with Hooks](#control-flow-with-hooks)
- [Advanced Agent Patterns](#advanced-agent-patterns)
- [Architecture Patterns](#architecture-patterns)
- [Best Practices](#best-practices)
- [Complete Examples](#complete-examples)

---

## Overview

### What is the Claude Agent SDK?

The Claude Agent SDK for Python enables developers to build agentic applications by providing programmatic access to Claude Code's powerful tool system. Unlike traditional LLM integrations that require you to implement tool execution yourself, this SDK leverages Claude Code's built-in tools (file operations, bash, git, grep, etc.) and allows you to add custom tools that run in-process.

### When to Use This SDK

**Use the Claude Agent SDK when you need:**

- **Autonomous Tool Use**: Claude can use many built-in tools (Read, Write, Bash, Git, Grep, Glob, etc.) without manual implementation
- **File System Operations**: Building agents that interact with codebases, projects, or file systems
- **Custom Tools**: Adding domain-specific capabilities as Python functions (in-process MCP servers)
- **Control & Safety**: Hooks and permission callbacks for fine-grained control over agent behavior
- **Interactive Sessions**: Multi-turn conversations with maintained context
- **Development Workflows**: Code review, testing, refactoring, documentation generation

**Don't use this SDK when you need:**

- Simple LLM API calls (use [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python))
- Custom tool execution logic (use [Anthropic Messages API](https://docs.anthropic.com/en/docs/build-with-claude/tool-use))
- Serverless deployments (requires Node.js runtime for Claude Code)
- Ultra-low latency (subprocess communication adds overhead)

### Key Differentiators

1. **Built-in Tool Ecosystem**: Many tools ready to use (file I/O, git, bash, grep, glob, etc.) - see [Claude Code tools documentation](https://docs.anthropic.com/en/docs/claude-code/settings#tools-available-to-claude)
2. **In-Process Custom Tools**: Define tools as Python functions, no subprocess management
3. **Bidirectional Control**: Hooks and callbacks for deterministic control flow
4. **Session Management**: Resume, fork, and manage conversation state
5. **Developer-Focused**: Built for agentic coding workflows

---

## Core Concepts

### The Two APIs: `query()` vs `ClaudeSDKClient`

The SDK provides two APIs optimized for different use cases:

#### `query()` - Unidirectional Streaming

**Best for:**
- One-off questions or tasks
- Batch processing
- Fire-and-forget operations
- When all inputs are known upfront

**Characteristics:**
- Async generator pattern
- Unidirectional: send inputs upfront, receive responses
- Stateless: each call is independent
- No interrupts or mid-stream control

**Example:**
```python
from claude_agent_sdk import query, ClaudeAgentOptions
from claude_agent_sdk.types import AssistantMessage, TextBlock

async for message in query(
    prompt="Analyze the code in src/ and suggest improvements",
    options=ClaudeAgentOptions(
        allowed_tools=["Read", "Grep", "Glob"],
        system_prompt="You are a code review assistant"
    )
):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if isinstance(block, TextBlock):
                print(block.text)
```

#### `ClaudeSDKClient` - Bidirectional Interactive Sessions

**Best for:**
- Chat interfaces
- Multi-turn conversations
- Interactive debugging
- When you need to respond to Claude's actions
- Real-time applications with user input

**Characteristics:**
- Bidirectional: send/receive messages at any time
- Stateful: maintains conversation context
- Interactive: send follow-ups based on responses
- Supports interrupts, permission controls, custom tools, and hooks

**Example:**
```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

async with ClaudeSDKClient() as client:
    # First query
    await client.query("What files are in this project?")
    async for msg in client.receive_response():
        display_message(msg)

    # Follow-up (context maintained)
    await client.query("Read the most important file")
    async for msg in client.receive_response():
        display_message(msg)

    # Dynamic permission control
    await client.set_permission_mode('acceptEdits')

    # Continue with auto-approved edits
    await client.query("Now refactor the main function")
    async for msg in client.receive_response():
        display_message(msg)
```

### Message Types

All messages flow through a strongly-typed system:

#### `UserMessage`
Content from the user or tool results
```python
@dataclass
class UserMessage:
    content: str | list[ContentBlock]
    parent_tool_use_id: str | None = None
```

#### `AssistantMessage`
Claude's responses and tool use requests
```python
@dataclass
class AssistantMessage:
    content: list[ContentBlock]  # TextBlock, ThinkingBlock, ToolUseBlock
    model: str
    parent_tool_use_id: str | None = None
```

#### `SystemMessage`
Metadata and system events
```python
@dataclass
class SystemMessage:
    subtype: str  # "init", "tool_use", etc.
    data: dict[str, Any]
```

#### `ResultMessage`
Session completion with cost/usage
```python
@dataclass
class ResultMessage:
    subtype: str
    duration_ms: int
    duration_api_ms: int
    is_error: bool
    num_turns: int
    session_id: str
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    result: str | None = None
```

#### `StreamEvent`
Partial message updates (when `include_partial_messages=True`)
```python
@dataclass
class StreamEvent:
    uuid: str
    session_id: str
    event: dict[str, Any]  # Raw Anthropic API event
    parent_tool_use_id: str | None = None
```

### Content Blocks

Messages contain typed content blocks:

```python
@dataclass
class TextBlock:
    text: str

@dataclass
class ThinkingBlock:
    thinking: str
    signature: str

@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]

@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str | list[dict[str, Any]] | None
    is_error: bool | None
```

---

## Getting Started

### Installation

```bash
# Install SDK
pip install claude-agent-sdk

# Install Claude Code (required)
npm install -g @anthropic-ai/claude-code

# Verify installation
claude --version  # Should be 2.0.0+
```

### Basic Query Pattern

```python
import anyio
from claude_agent_sdk import query, ClaudeAgentOptions

async def main():
    async for message in query(
        prompt="What is 2 + 2?",
        options=ClaudeAgentOptions(max_turns=1)
    ):
        print(message)

anyio.run(main)
```

### Interactive Session Pattern

```python
import anyio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

async def main():
    async with ClaudeSDKClient() as client:
        await client.query("Hello Claude!")

        async for msg in client.receive_response():
            print(msg)

anyio.run(main)
```

### Configuration Options

```python
from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(
    # Tools
    allowed_tools=["Read", "Write", "Bash", "Grep"],
    disallowed_tools=["Task"],  # Prevent subagents

    # System prompt
    system_prompt="You are a helpful coding assistant",

    # Or use preset with append
    system_prompt={
        "type": "preset",
        "preset": "claude_code",
        "append": "Always explain your reasoning"
    },

    # Session control
    max_turns=10,
    resume="session-id-123",  # Resume previous session
    fork_session=True,  # Branch from resumed session

    # Permissions
    permission_mode="acceptEdits",  # or "default", "plan", "bypassPermissions"

    # Environment
    cwd="/path/to/project",
    add_dirs=["/additional/context"],
    env={"CUSTOM_VAR": "value"},

    # Advanced
    include_partial_messages=True,  # Stream partial updates
    setting_sources=["user", "project"],  # Load settings
)
```

---

## Building Custom Tools

### SDK MCP Servers: In-Process Custom Tools

The SDK supports defining custom tools as Python functions that run in your application's process. These are implemented as in-process MCP (Model Context Protocol) servers, eliminating subprocess overhead.

### Benefits Over External MCP Servers

| Feature | SDK MCP (In-Process) | External MCP (Subprocess) |
|---------|---------------------|---------------------------|
| **Performance** | Direct function calls | IPC overhead |
| **Deployment** | Single Python process | Multiple processes |
| **Debugging** | Standard Python debugging | Inter-process debugging |
| **State Access** | Direct access to app state | No app state access |
| **Setup** | Simple decorator | Separate server process |

### Defining a Tool

Use the `@tool` decorator to create tools:

```python
from claude_agent_sdk import tool
from typing import Any

@tool(
    "calculate_distance",
    "Calculate distance between two points",
    {"x1": float, "y1": float, "x2": float, "y2": float}
)
async def calculate_distance(args: dict[str, Any]) -> dict[str, Any]:
    """Calculate Euclidean distance between two points."""
    import math

    distance = math.sqrt(
        (args["x2"] - args["x1"]) ** 2 +
        (args["y2"] - args["y1"]) ** 2
    )

    return {
        "content": [
            {
                "type": "text",
                "text": f"Distance: {distance:.2f} units"
            }
        ]
    }
```

### Input Schema Formats

**Simple type mapping:**
```python
input_schema = {"name": str, "age": int, "active": bool}
# Automatically converts to JSON Schema:
# {"type": "object", "properties": {"name": {"type": "string"}, ...}}
```

**Full JSON Schema:**
```python
input_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0, "maximum": 150}
    },
    "required": ["name"]
}
```

### Error Handling in Tools

Tools can return errors gracefully:

```python
@tool("divide", "Divide two numbers", {"a": float, "b": float})
async def divide(args: dict[str, Any]) -> dict[str, Any]:
    if args["b"] == 0:
        return {
            "content": [
                {"type": "text", "text": "Error: Division by zero"}
            ],
            "is_error": True
        }

    result = args["a"] / args["b"]
    return {
        "content": [
            {"type": "text", "text": f"Result: {result}"}
        ]
    }
```

### Creating and Using SDK MCP Servers

```python
from claude_agent_sdk import create_sdk_mcp_server, ClaudeSDKClient, ClaudeAgentOptions

# Step 1: Define tools
@tool("greet", "Greet a user", {"name": str})
async def greet(args):
    return {"content": [{"type": "text", "text": f"Hello, {args['name']}!"}]}

@tool("farewell", "Say goodbye", {"name": str})
async def farewell(args):
    return {"content": [{"type": "text", "text": f"Goodbye, {args['name']}!"}]}

# Step 2: Create SDK MCP server
server = create_sdk_mcp_server(
    name="greetings",
    version="1.0.0",
    tools=[greet, farewell]
)

# Step 3: Configure and use
options = ClaudeAgentOptions(
    mcp_servers={"greetings": server},
    allowed_tools=[
        "mcp__greetings__greet",
        "mcp__greetings__farewell"
    ]
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("Greet Alice and then say goodbye to Bob")
    async for msg in client.receive_response():
        print(msg)
```

### Tool Naming Convention

SDK MCP tools follow the naming pattern:
```
mcp__{server_name}__{tool_name}
```

Examples:
- Server "calc" with tool "add" → `"mcp__calc__add"`
- Server "db" with tool "query" → `"mcp__db__query"`

### Complete Calculator Example

```python
from claude_agent_sdk import tool, create_sdk_mcp_server, ClaudeSDKClient, ClaudeAgentOptions
from typing import Any

@tool("add", "Add two numbers", {"a": float, "b": float})
async def add_numbers(args: dict[str, Any]) -> dict[str, Any]:
    result = args["a"] + args["b"]
    return {"content": [{"type": "text", "text": f"{args['a']} + {args['b']} = {result}"}]}

@tool("subtract", "Subtract two numbers", {"a": float, "b": float})
async def subtract_numbers(args: dict[str, Any]) -> dict[str, Any]:
    result = args["a"] - args["b"]
    return {"content": [{"type": "text", "text": f"{args['a']} - {args['b']} = {result}"}]}

@tool("multiply", "Multiply two numbers", {"a": float, "b": float})
async def multiply_numbers(args: dict[str, Any]) -> dict[str, Any]:
    result = args["a"] * args["b"]
    return {"content": [{"type": "text", "text": f"{args['a']} × {args['b']} = {result}"}]}

# Create server
calculator = create_sdk_mcp_server(
    name="calculator",
    version="1.0.0",
    tools=[add_numbers, subtract_numbers, multiply_numbers]
)

# Use in agent
async def main():
    options = ClaudeAgentOptions(
        mcp_servers={"calc": calculator},
        allowed_tools=[
            "mcp__calc__add",
            "mcp__calc__subtract",
            "mcp__calc__multiply"
        ]
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("Calculate (15 + 27) * 3")
        async for msg in client.receive_response():
            print(msg)

import anyio
anyio.run(main)
```

### Mixing SDK and External MCP Servers

You can use both SDK (in-process) and external (subprocess) MCP servers together:

```python
options = ClaudeAgentOptions(
    mcp_servers={
        # SDK MCP server (in-process)
        "internal": create_sdk_mcp_server(
            name="internal",
            tools=[my_tool_function]
        ),

        # External MCP server (subprocess)
        "external": {
            "type": "stdio",
            "command": "python",
            "args": ["-m", "external_server"]
        },

        # SSE MCP server (HTTP)
        "remote": {
            "type": "sse",
            "url": "https://api.example.com/mcp",
            "headers": {"Authorization": "Bearer token"}
        }
    },
    allowed_tools=[
        "mcp__internal__my_tool",
        "mcp__external__external_tool",
        "mcp__remote__remote_tool"
    ]
)
```

---

## Control Flow with Hooks

Hooks provide deterministic control over Claude's execution lifecycle. They're callback functions that the Claude Code application (not Claude itself) invokes at specific points.

### Available Hook Events

| Hook Event | When Fired | Use Cases |
|-----------|------------|-----------|
| `PreToolUse` | Before tool execution | Permission checks, input validation, logging |
| `PostToolUse` | After tool execution | Result analysis, context injection, error handling |
| `UserPromptSubmit` | When user submits prompt | Adding context, logging, preprocessing |
| `Stop` | When execution stops | Cleanup, reporting, state saving |
| `SubagentStop` | When subagent stops | Subagent tracking, result aggregation |
| `PreCompact` | Before context compaction | Custom compaction logic, state preservation |

**Note**: The Python SDK does not support `SessionStart`, `SessionEnd`, and `Notification` hooks due to setup limitations. However, you can use `UserPromptSubmit` hook with `SessionStart` in `hookSpecificOutput` for similar functionality.

### Hook Callback Signature

```python
from claude_agent_sdk import HookInput, HookContext, HookJSONOutput

async def my_hook(
    input_data: HookInput,        # Strongly-typed hook input
    tool_use_id: str | None,      # Optional tool identifier
    context: HookContext          # Hook context (signal, etc.)
) -> HookJSONOutput:              # Hook output (sync or async)
    # Your logic here
    return {}
```

### Hook Input Types

Hook inputs are discriminated unions based on `hook_event_name`:

```python
# PreToolUse
{
    "hook_event_name": "PreToolUse",
    "tool_name": "Bash",
    "tool_input": {"command": "ls -la"},
    "session_id": "...",
    "transcript_path": "...",
    "cwd": "..."
}

# PostToolUse
{
    "hook_event_name": "PostToolUse",
    "tool_name": "Bash",
    "tool_input": {...},
    "tool_response": "output here",
    ...
}

# UserPromptSubmit
{
    "hook_event_name": "UserPromptSubmit",
    "prompt": "What is 2+2?",
    ...
}
```

### Hook Output Types

**Synchronous (immediate) response:**
```python
{
    # Control fields
    "continue_": True,  # Continue execution (default: True)
    "suppressOutput": False,  # Hide stdout (default: False)
    "stopReason": "Reason for stopping",  # Required if continue_ is False

    # Decision fields
    "decision": "block",  # Block execution
    "systemMessage": "Warning message for user",
    "reason": "Explanation for Claude",

    # Hook-specific output
    "hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": "allow",  # or "deny", "ask"
        "permissionDecisionReason": "Explanation",
        "updatedInput": {...}  # Modified tool input
    }
}
```

**Asynchronous (deferred) response:**
```python
{
    "async_": True,  # Defer execution
    "asyncTimeout": 5000  # Timeout in milliseconds
}
```

**Note**: Use `async_` and `continue_` (with underscores) in Python code. The SDK automatically converts them to `async` and `continue` for the CLI.

### Example 1: PreToolUse - Block Dangerous Commands

```python
from claude_agent_sdk import HookMatcher, ClaudeAgentOptions

async def check_bash_command(input_data, tool_use_id, context):
    """Block dangerous bash commands."""
    if input_data["tool_name"] != "Bash":
        return {}

    command = input_data["tool_input"].get("command", "")
    dangerous = ["rm -rf /", "sudo", "dd if="]

    for pattern in dangerous:
        if pattern in command:
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": f"Dangerous pattern: {pattern}"
                }
            }

    return {}

options = ClaudeAgentOptions(
    allowed_tools=["Bash"],
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[check_bash_command])
        ]
    }
)
```

### Example 2: PostToolUse - Add Context on Errors

```python
async def review_tool_output(input_data, tool_use_id, context):
    """Add context when tools error."""
    tool_response = str(input_data.get("tool_response", ""))

    if "error" in tool_response.lower():
        return {
            "systemMessage": "⚠️ Command produced an error",
            "reason": "Tool execution failed - consider checking syntax",
            "hookSpecificOutput": {
                "hookEventName": "PostToolUse",
                "additionalContext": "The command encountered an error. Review the output carefully."
            }
        }

    return {}

options = ClaudeAgentOptions(
    hooks={
        "PostToolUse": [
            HookMatcher(matcher="Bash", hooks=[review_tool_output])
        ]
    }
)
```

### Example 3: Stop Execution on Critical Errors

```python
async def stop_on_critical_error(input_data, tool_use_id, context):
    """Stop execution if critical error detected."""
    tool_response = str(input_data.get("tool_response", ""))

    if "critical" in tool_response.lower():
        return {
            "continue_": False,
            "stopReason": "Critical error detected - execution halted",
            "systemMessage": "🛑 Execution stopped due to critical error"
        }

    return {"continue_": True}

options = ClaudeAgentOptions(
    hooks={
        "PostToolUse": [
            HookMatcher(matcher=None, hooks=[stop_on_critical_error])
        ]
    }
)
```

### Example 4: Inject Context on User Prompts

```python
async def add_custom_instructions(input_data, tool_use_id, context):
    """Add context to every user prompt."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "Remember: always explain your reasoning step by step"
        }
    }

options = ClaudeAgentOptions(
    hooks={
        "UserPromptSubmit": [
            HookMatcher(matcher=None, hooks=[add_custom_instructions])
        ]
    }
)
```

### HookMatcher Configuration

The `matcher` field filters which events trigger the hooks:

```python
from claude_agent_sdk import HookMatcher

# Match specific tool
HookMatcher(matcher="Bash", hooks=[...])

# Match multiple tools (regex-like)
HookMatcher(matcher="Write|Edit|MultiEdit", hooks=[...])

# Match all events of this type
HookMatcher(matcher=None, hooks=[...])

# Multiple matchers for same event
options = ClaudeAgentOptions(
    hooks={
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[bash_hook]),
            HookMatcher(matcher="Write", hooks=[write_hook]),
            HookMatcher(matcher=None, hooks=[global_hook])
        ]
    }
)
```

---

## Advanced Agent Patterns

### Pattern 1: Tool Permission Callbacks

For fine-grained control over individual tool invocations, use the `can_use_tool` callback:

```python
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    ToolPermissionContext,
    PermissionResultAllow,
    PermissionResultDeny
)

async def permission_callback(
    tool_name: str,
    input_data: dict,
    context: ToolPermissionContext
) -> PermissionResultAllow | PermissionResultDeny:
    """Control tool permissions dynamically."""

    # Always allow read operations
    if tool_name in ["Read", "Glob", "Grep"]:
        return PermissionResultAllow()

    # Deny system directory writes
    if tool_name in ["Write", "Edit"]:
        file_path = input_data.get("file_path", "")
        if file_path.startswith("/etc/") or file_path.startswith("/usr/"):
            return PermissionResultDeny(
                message=f"Cannot modify system directory: {file_path}"
            )

    # Modify input to redirect to safe location
    if tool_name == "Write":
        file_path = input_data.get("file_path", "")
        if not file_path.startswith("./safe/"):
            modified = input_data.copy()
            modified["file_path"] = f"./safe/{file_path.split('/')[-1]}"
            return PermissionResultAllow(updated_input=modified)

    # Block dangerous bash commands
    if tool_name == "Bash":
        command = input_data.get("command", "")
        if any(danger in command for danger in ["rm -rf", "sudo", "mkfs"]):
            return PermissionResultDeny(
                message="Dangerous command blocked"
            )

    # Default: allow
    return PermissionResultAllow()

options = ClaudeAgentOptions(
    can_use_tool=permission_callback,
    permission_mode="default"
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("Create and modify some files")
    async for msg in client.receive_response():
        print(msg)
```

**Key differences from hooks:**
- `can_use_tool`: Per-invocation control, can modify inputs
- Hooks: Broader lifecycle control, can add context or stop execution

### Pattern 2: Dynamic Permission Modes

Change permission modes during a conversation:

```python
async with ClaudeSDKClient() as client:
    # Start in review mode (default permissions)
    await client.query("Analyze this codebase and suggest changes")
    async for msg in client.receive_response():
        display(msg)

    # User reviews suggestions, then enables auto-accept
    await client.set_permission_mode('acceptEdits')

    # Claude can now make edits without prompting
    await client.query("Implement the changes we discussed")
    async for msg in client.receive_response():
        display(msg)

    # Switch back to careful mode for final review
    await client.set_permission_mode('default')
    await client.query("Review what you changed")
    async for msg in client.receive_response():
        display(msg)
```

### Pattern 3: Session Forking for Exploration

Resume a session and fork to explore different approaches:

```python
# Initial session
async with ClaudeSDKClient() as client:
    await client.query("Design a REST API for a todo app")
    async for msg in client.receive_response():
        if isinstance(msg, ResultMessage):
            session_id = msg.session_id
            print(f"Session ID: {session_id}")

# Fork 1: Explore SQL database approach
async with ClaudeSDKClient(options=ClaudeAgentOptions(
    resume=session_id,
    fork_session=True
)) as client:
    await client.query("Implement with PostgreSQL")
    async for msg in client.receive_response():
        display(msg)

# Fork 2: Explore NoSQL approach (from same starting point)
async with ClaudeSDKClient(options=ClaudeAgentOptions(
    resume=session_id,
    fork_session=True
)) as client:
    await client.query("Implement with MongoDB")
    async for msg in client.receive_response():
        display(msg)
```

### Pattern 4: Programmatic Subagents

Define custom agents with specific tools and prompts:

```python
from claude_agent_sdk import AgentDefinition

options = ClaudeAgentOptions(
    agents={
        "code-reviewer": AgentDefinition(
            description="Reviews code for best practices",
            prompt="You are a code reviewer. Focus on bugs, security, and performance.",
            tools=["Read", "Grep", "Glob"],
            model="sonnet"
        ),
        "test-writer": AgentDefinition(
            description="Writes comprehensive tests",
            prompt="You are a testing expert. Write thorough, maintainable tests.",
            tools=["Read", "Write", "Bash"],
            model="sonnet"
        ),
        "docs-generator": AgentDefinition(
            description="Generates documentation",
            prompt="You create clear, comprehensive documentation.",
            tools=["Read", "Write", "Glob"],
            model="sonnet"
        )
    }
)

async with ClaudeSDKClient(options=options) as client:
    # Use code-reviewer agent
    await client.query("Use the code-reviewer agent to review src/main.py")
    async for msg in client.receive_response():
        display(msg)

    # Switch to test-writer agent
    await client.query("Use the test-writer agent to create tests for main.py")
    async for msg in client.receive_response():
        display(msg)
```

### Pattern 5: Partial Message Streaming for Real-Time UIs

Build responsive UIs by streaming partial updates:

```python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, ResultMessage
from claude_agent_sdk.types import StreamEvent

options = ClaudeAgentOptions(
    include_partial_messages=True
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("Write a long analysis of this codebase")

    current_text = ""
    async for msg in client.receive_messages():
        if isinstance(msg, StreamEvent):
            # Handle partial updates
            event = msg.event
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text_chunk = delta.get("text", "")
                    current_text += text_chunk
                    update_ui(current_text)  # Real-time UI update

        elif isinstance(msg, ResultMessage):
            break
```

### Pattern 6: Multi-Session Orchestration

Manage multiple concurrent sessions:

```python
import anyio

async def run_agent_session(session_id: str, task: str):
    """Run an agent session with a specific task."""
    async with ClaudeSDKClient() as client:
        await client.query(task)

        results = []
        async for msg in client.receive_response():
            if isinstance(msg, AssistantMessage):
                results.append(msg)

        return {"session_id": session_id, "results": results}

async def main():
    # Run multiple agents in parallel
    async with anyio.create_task_group() as tg:
        task1 = tg.start_soon(
            run_agent_session,
            "code-review",
            "Review the code in src/"
        )
        task2 = tg.start_soon(
            run_agent_session,
            "test-check",
            "Verify all tests pass"
        )
        task3 = tg.start_soon(
            run_agent_session,
            "doc-check",
            "Check documentation completeness"
        )

    print("All agent sessions completed")

anyio.run(main)
```

### Pattern 7: State Management with Context

Maintain application state across agent interactions:

```python
class AgentState:
    def __init__(self):
        self.files_modified = []
        self.tests_run = []
        self.errors_found = []

state = AgentState()

async def track_file_modifications(input_data, tool_use_id, context):
    """Track file modifications in application state."""
    if input_data["tool_name"] in ["Write", "Edit", "MultiEdit"]:
        file_path = input_data["tool_input"].get("file_path")
        if file_path:
            state.files_modified.append(file_path)
    return {}

async def track_test_execution(input_data, tool_use_id, context):
    """Track test execution results."""
    if input_data["tool_name"] == "Bash":
        command = input_data["tool_input"].get("command", "")
        if "pytest" in command or "unittest" in command:
            response = input_data.get("tool_response", "")
            state.tests_run.append({
                "command": command,
                "output": response,
                "success": "error" not in response.lower()
            })
    return {}

options = ClaudeAgentOptions(
    hooks={
        "PostToolUse": [
            HookMatcher(
                matcher="Write|Edit|MultiEdit",
                hooks=[track_file_modifications]
            ),
            HookMatcher(
                matcher="Bash",
                hooks=[track_test_execution]
            )
        ]
    }
)

async with ClaudeSDKClient(options=options) as client:
    await client.query("Refactor main.py and run tests")
    async for msg in client.receive_response():
        pass

    # Access state after completion
    print(f"Modified files: {state.files_modified}")
    print(f"Tests run: {len(state.tests_run)}")
    print(f"Test success rate: {sum(t['success'] for t in state.tests_run) / len(state.tests_run)}")
```

---

## Architecture Patterns

### Pattern 1: Agent Supervisor Architecture

Create a supervisor agent that delegates to specialized agents:

```python
from claude_agent_sdk import AgentDefinition, ClaudeSDKClient, ClaudeAgentOptions

class AgentSupervisor:
    def __init__(self):
        self.options = ClaudeAgentOptions(
            agents={
                "analyzer": AgentDefinition(
                    description="Analyzes code structure and patterns",
                    prompt="Analyze code architecture and patterns.",
                    tools=["Read", "Grep", "Glob"]
                ),
                "implementer": AgentDefinition(
                    description="Implements code changes",
                    prompt="Implement code changes carefully.",
                    tools=["Read", "Write", "Edit", "Bash"]
                ),
                "tester": AgentDefinition(
                    description="Tests code changes",
                    prompt="Run and verify tests.",
                    tools=["Bash", "Read"]
                )
            }
        )

    async def run_workflow(self, task: str):
        """Run a complete workflow with multiple agents."""
        async with ClaudeSDKClient(options=self.options) as client:
            # Phase 1: Analysis
            await client.query(
                f"Use the analyzer agent to analyze the task: {task}"
            )
            analysis_results = []
            async for msg in client.receive_response():
                analysis_results.append(msg)

            # Phase 2: Implementation
            await client.query(
                "Use the implementer agent to implement based on the analysis"
            )
            async for msg in client.receive_response():
                pass

            # Phase 3: Testing
            await client.query(
                "Use the tester agent to run all tests and verify"
            )
            test_results = []
            async for msg in client.receive_response():
                test_results.append(msg)

            return {
                "analysis": analysis_results,
                "tests": test_results
            }

# Usage
supervisor = AgentSupervisor()
results = await supervisor.run_workflow("Add user authentication")
```

### Pattern 2: Tool-First Architecture

Build agents around custom domain-specific tools:

```python
# Step 1: Define domain-specific tools
@tool("query_database", "Query the application database", {
    "query": str,
    "limit": int
})
async def query_database(args):
    # Your database logic
    results = await db.execute(args["query"], limit=args["limit"])
    return {"content": [{"type": "text", "text": str(results)}]}

@tool("validate_schema", "Validate data against schema", {
    "data": dict,
    "schema_name": str
})
async def validate_schema(args):
    # Your validation logic
    is_valid = await validator.validate(args["data"], args["schema_name"])
    return {"content": [{"type": "text", "text": f"Valid: {is_valid}"}]}

@tool("deploy_change", "Deploy a change to production", {
    "change_id": str,
    "environment": str
})
async def deploy_change(args):
    # Your deployment logic
    result = await deployer.deploy(args["change_id"], args["environment"])
    return {"content": [{"type": "text", "text": f"Deployed: {result}"}]}

# Step 2: Create SDK MCP server
domain_server = create_sdk_mcp_server(
    name="domain",
    tools=[query_database, validate_schema, deploy_change]
)

# Step 3: Build agent around tools
class DomainAgent:
    def __init__(self):
        self.options = ClaudeAgentOptions(
            mcp_servers={"domain": domain_server},
            allowed_tools=[
                "mcp__domain__query_database",
                "mcp__domain__validate_schema",
                "mcp__domain__deploy_change",
                # Also allow built-in tools
                "Read", "Write"
            ],
            system_prompt="You are a domain expert. Use domain tools to solve problems."
        )

    async def execute(self, task: str):
        async with ClaudeSDKClient(options=self.options) as client:
            await client.query(task)
            async for msg in client.receive_response():
                yield msg

# Usage
agent = DomainAgent()
async for result in agent.execute("Query users and validate their data"):
    print(result)
```

### Pattern 3: Pipeline Architecture

Chain multiple agent steps with validation:

```python
class AgentPipeline:
    def __init__(self):
        self.steps = []
        self.results = {}

    def add_step(self, name: str, agent_options: ClaudeAgentOptions, prompt: str):
        """Add a step to the pipeline."""
        self.steps.append({
            "name": name,
            "options": agent_options,
            "prompt": prompt
        })

    async def run(self):
        """Execute all pipeline steps."""
        for step in self.steps:
            print(f"Running step: {step['name']}")

            async with ClaudeSDKClient(options=step['options']) as client:
                # Inject results from previous steps
                context = "\n".join([
                    f"Results from {name}: {result}"
                    for name, result in self.results.items()
                ])

                full_prompt = f"{context}\n\n{step['prompt']}"
                await client.query(full_prompt)

                step_results = []
                async for msg in client.receive_response():
                    step_results.append(msg)

                self.results[step['name']] = step_results

        return self.results

# Usage
pipeline = AgentPipeline()

pipeline.add_step(
    "analyze",
    ClaudeAgentOptions(
        allowed_tools=["Read", "Grep"],
        system_prompt="Analyze code structure"
    ),
    "Analyze the codebase in src/"
)

pipeline.add_step(
    "implement",
    ClaudeAgentOptions(
        allowed_tools=["Read", "Write", "Edit"],
        system_prompt="Implement changes carefully"
    ),
    "Implement improvements based on the analysis"
)

pipeline.add_step(
    "test",
    ClaudeAgentOptions(
        allowed_tools=["Bash", "Read"],
        system_prompt="Verify all tests pass"
    ),
    "Run tests and verify everything works"
)

results = await pipeline.run()
```

### Pattern 4: Feedback Loop Architecture

Create agents that learn from execution results:

```python
class AdaptiveAgent:
    def __init__(self):
        self.execution_history = []
        self.learned_patterns = []

    async def execute_with_retry(self, task: str, max_attempts: int = 3):
        """Execute task with automatic retry and learning."""

        for attempt in range(max_attempts):
            # Build context from previous attempts
            retry_context = ""
            if attempt > 0:
                retry_context = f"""
                Previous attempt failed. Learn from these errors:
                {self._format_previous_attempts()}
                """

            async with ClaudeSDKClient(options=self._build_options()) as client:
                full_prompt = f"{retry_context}\n{task}"
                await client.query(full_prompt)

                execution_log = {
                    "attempt": attempt,
                    "errors": [],
                    "success": False
                }

                async for msg in client.receive_response():
                    # Track errors
                    if isinstance(msg, UserMessage):
                        for block in msg.content if isinstance(msg.content, list) else []:
                            if isinstance(block, ToolResultBlock) and block.is_error:
                                execution_log["errors"].append(block.content)

                    # Check for success
                    if isinstance(msg, ResultMessage):
                        execution_log["success"] = not msg.is_error

                self.execution_history.append(execution_log)

                if execution_log["success"]:
                    self._learn_from_success(execution_log)
                    return execution_log
                else:
                    self._learn_from_failure(execution_log)

        raise Exception(f"Failed after {max_attempts} attempts")

    def _format_previous_attempts(self):
        return "\n".join([
            f"Attempt {log['attempt']}: {', '.join(log['errors'])}"
            for log in self.execution_history
        ])

    def _learn_from_success(self, execution_log):
        # Extract successful patterns
        pass

    def _learn_from_failure(self, execution_log):
        # Learn from errors
        pass

    def _build_options(self):
        # Build options with learned patterns
        return ClaudeAgentOptions(
            system_prompt=f"Avoid these patterns: {self.learned_patterns}",
            allowed_tools=["Read", "Write", "Bash"]
        )

# Usage
agent = AdaptiveAgent()
result = await agent.execute_with_retry("Refactor the authentication module")
```

---

## Best Practices

### 1. Tool Configuration

**✅ DO:**
- Pre-approve read-only tools: `allowed_tools=["Read", "Grep", "Glob"]`
- Use `disallowed_tools` to explicitly block dangerous operations
- Start restrictive, then gradually enable tools as needed
- Use `permission_mode="acceptEdits"` for trusted automation

**❌ DON'T:**
- Use `permission_mode="bypassPermissions"` in production
- Allow all tools by default
- Mix incompatible permission modes (e.g., `can_use_tool` with `permission_prompt_tool_name`)

### 2. Error Handling

**✅ DO:**
```python
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    ClaudeSDKError,
    CLINotFoundError,
    ProcessError,
    CLIConnectionError,
    CLIJSONDecodeError
)

try:
    async with ClaudeSDKClient(options) as client:
        await client.query("Task")
        async for msg in client.receive_response():
            process_message(msg)
except CLINotFoundError:
    print("Install Claude Code: npm install -g @anthropic-ai/claude-code")
except ProcessError as e:
    print(f"CLI process failed (exit {e.exit_code}): {e.stderr}")
except CLIConnectionError as e:
    print(f"Connection error: {e}")
except CLIJSONDecodeError as e:
    print(f"JSON decode error: {e.line[:100]}")
except ClaudeSDKError as e:
    print(f"SDK error: {e}")
```

**❌ DON'T:**
- Catch all exceptions without specific handling
- Ignore errors in hooks or tool callbacks
- Let exceptions propagate silently

### 3. Session Management

**✅ DO:**
- Use context managers (`async with`) for automatic cleanup
- Store session IDs for resumption: `session_id = result_msg.session_id`
- Use `fork_session=True` to explore alternatives without losing history
- Set reasonable `max_turns` limits to prevent runaway execution

**❌ DON'T:**
- Forget to call `disconnect()` when not using context manager
- Resume sessions without checking if they still exist
- Use extremely high `max_turns` without monitoring

### 4. Custom Tools

**✅ DO:**
- Use descriptive tool names and descriptions
- Provide detailed JSON schemas for complex inputs
- Handle errors gracefully and return `is_error: True`
- Make tools async (use `async def`)
- Keep tools focused on single responsibilities

**❌ DON'T:**
- Make tools that block (use async I/O)
- Return unstructured data (always use MCP content format)
- Create tools with overlapping functionality
- Forget to add tools to `allowed_tools`

### 5. Hooks

**✅ DO:**
- Use hooks for deterministic control flow
- Make hooks fast (they block execution)
- Use `HookMatcher` to target specific tools
- Return empty dict `{}` when no action needed
- Use `async_: True` for expensive operations

**❌ DON'T:**
- Put slow operations in synchronous hooks
- Raise exceptions from hooks (return error responses instead)
- Use hooks for every event (only when needed)
- Forget to handle different hook input types

### 6. Testing

**✅ DO:**
```python
import pytest
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, ResultMessage

@pytest.mark.asyncio
async def test_agent_behavior():
    """Test agent with controlled environment."""
    options = ClaudeAgentOptions(
        allowed_tools=["Read"],
        max_turns=1,
        cwd="/tmp/test-workspace"
    )

    async with ClaudeSDKClient(options=options) as client:
        await client.query("Read test_file.txt")

        messages = []
        async for msg in client.receive_response():
            messages.append(msg)

        # Assert expected behavior
        assert len(messages) > 0
        assert any(isinstance(msg, ResultMessage) for msg in messages)
```

**❌ DON'T:**
- Test against production systems
- Forget to mock external dependencies
- Skip testing error paths
- Leave test artifacts in file system

### 7. Performance

**✅ DO:**
- Use `query()` for simple, one-off operations
- Use `ClaudeSDKClient` for complex interactions
- Enable `include_partial_messages` for responsive UIs
- Run independent agent sessions in parallel
- Set appropriate `max_buffer_size` for large outputs

**❌ DON'T:**
- Create new client for every query (reuse when possible)
- Process large files without streaming
- Block on synchronous operations in hooks/callbacks
- Forget to clean up resources

### 8. Security

**✅ DO:**
- Use `can_use_tool` callback for sensitive operations
- Validate all tool inputs in hooks
- Use `add_dirs` to restrict file system access
- Set working directory with `cwd` option
- Use hooks to log all tool usage

**❌ DON'T:**
- Use `permission_mode="bypassPermissions"` without review
- Trust user input without validation
- Allow write access to system directories
- Expose sensitive data in tool outputs

---

## Complete Examples

### Example 1: Code Review Bot

```python
import anyio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AgentDefinition
from claude_agent_sdk.types import AssistantMessage, TextBlock, ToolResultBlock

class CodeReviewBot:
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.options = ClaudeAgentOptions(
            cwd=repo_path,
            allowed_tools=["Read", "Grep", "Glob", "Bash"],
            agents={
                "reviewer": AgentDefinition(
                    description="Code review specialist",
                    prompt="""You are a senior code reviewer. Focus on:
                    - Code quality and maintainability
                    - Security vulnerabilities
                    - Performance issues
                    - Best practices
                    Provide constructive, actionable feedback.""",
                    tools=["Read", "Grep", "Glob"],
                    model="sonnet"
                )
            }
        )

    async def review_file(self, file_path: str) -> dict:
        """Review a single file."""
        async with ClaudeSDKClient(options=self.options) as client:
            await client.query(
                f"Use the reviewer agent to review {file_path}. "
                "Provide a summary of issues and suggestions."
            )

            review_results = {
                "file": file_path,
                "issues": [],
                "suggestions": []
            }

            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            # Parse review content
                            if "issue:" in block.text.lower():
                                review_results["issues"].append(block.text)
                            elif "suggest" in block.text.lower():
                                review_results["suggestions"].append(block.text)

            return review_results

    async def review_repository(self) -> list[dict]:
        """Review all Python files in repository."""
        # First, discover all Python files
        async with ClaudeSDKClient(options=self.options) as client:
            await client.query("Find all Python files in this repository")

            files = []
            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, ToolResultBlock):
                            # Extract file paths from tool results
                            files.extend(self._parse_file_paths(block.content))

        # Review each file
        reviews = []
        for file_path in files:
            review = await self.review_file(file_path)
            reviews.append(review)

        return reviews

    def _parse_file_paths(self, content: str | list[dict] | None) -> list[str]:
        """Extract file paths from tool output."""
        if not content or isinstance(content, list):
            return []
        # Implementation depends on tool output format
        return [line.strip() for line in content.split('\n') if line.endswith('.py')]

# Usage
async def main():
    bot = CodeReviewBot("/path/to/repo")
    reviews = await bot.review_repository()

    for review in reviews:
        print(f"\n=== {review['file']} ===")
        print(f"Issues: {len(review['issues'])}")
        for issue in review['issues']:
            print(f"  - {issue}")

anyio.run(main)
```

### Example 2: Automated Testing Agent

```python
import anyio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, HookMatcher, ResultMessage
from claude_agent_sdk.types import HookInput, HookContext, HookJSONOutput, AssistantMessage, ToolUseBlock

class TestingAgent:
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.test_results = []

        # Hook to capture test results
        async def capture_test_output(
            input_data: HookInput,
            tool_use_id: str | None,
            context: HookContext
        ) -> HookJSONOutput:
            # Only process PostToolUse events
            if input_data["hook_event_name"] != "PostToolUse":
                return {}

            if input_data["tool_name"] == "Bash":
                command = input_data["tool_input"].get("command", "")
                if "pytest" in command or "unittest" in command:
                    output = input_data.get("tool_response", "")
                    self.test_results.append({
                        "command": command,
                        "output": str(output),
                        "success": "FAILED" not in str(output)
                    })
            return {}

        self.options = ClaudeAgentOptions(
            cwd=project_path,
            allowed_tools=["Read", "Write", "Bash", "Grep"],
            permission_mode="acceptEdits",
            hooks={
                "PostToolUse": [
                    HookMatcher(matcher="Bash", hooks=[capture_test_output])
                ]
            }
        )

    async def run_all_tests(self) -> list[dict]:
        """Run all tests in the project."""
        async with ClaudeSDKClient(options=self.options) as client:
            await client.query(
                "Find all test files and run them. "
                "Report which tests passed and which failed."
            )

            async for msg in client.receive_response():
                pass  # Results captured by hook

            return self.test_results

    async def fix_failing_tests(self) -> list[str]:
        """Attempt to fix failing tests."""
        fixed_tests = []

        for result in self.test_results:
            if not result["success"]:
                async with ClaudeSDKClient(options=self.options) as client:
                    await client.query(
                        f"This test failed:\n{result['command']}\n\n"
                        f"Output:\n{result['output']}\n\n"
                        "Analyze the failure and attempt to fix it."
                    )

                    async for msg in client.receive_response():
                        if isinstance(msg, ResultMessage) and not msg.is_error:
                            fixed_tests.append(result["command"])

        return fixed_tests

    async def generate_missing_tests(self, file_path: str):
        """Generate tests for untested code."""
        async with ClaudeSDKClient(options=self.options) as client:
            await client.query(
                f"Analyze {file_path} and identify functions without tests. "
                f"Generate comprehensive tests for them."
            )

            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock) and block.name == "Write":
                            print(f"Generated test file: {block.input.get('file_path')}")

# Usage
async def main():
    agent = TestingAgent("/path/to/project")

    # Run all tests
    results = await agent.run_all_tests()
    print(f"Ran {len(results)} test suites")

    failing = [r for r in results if not r["success"]]
    if failing:
        print(f"\n{len(failing)} test suites failed. Attempting to fix...")
        fixed = await agent.fix_failing_tests()
        print(f"Fixed {len(fixed)} test suites")

    # Generate missing tests
    await agent.generate_missing_tests("src/main.py")

anyio.run(main)
```

### Example 3: Documentation Generator

```python
import anyio
from typing import Any
from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    tool,
    create_sdk_mcp_server
)
from claude_agent_sdk.types import AssistantMessage, TextBlock

# Custom tool to extract code structure
@tool("extract_structure", "Extract code structure from a file", {
    "file_path": str
})
async def extract_structure(args: dict[str, Any]) -> dict[str, Any]:
    """Extract functions, classes, and methods from a Python file."""
    import ast

    with open(args["file_path"]) as f:
        tree = ast.parse(f.read())

    structure = {
        "classes": [],
        "functions": []
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
            structure["classes"].append({
                "name": node.name,
                "methods": methods
            })
        elif isinstance(node, ast.FunctionDef):
            structure["functions"].append(node.name)

    return {
        "content": [
            {"type": "text", "text": str(structure)}
        ]
    }

class DocumentationGenerator:
    def __init__(self, project_path: str):
        # Create SDK MCP server with custom tool
        structure_server = create_sdk_mcp_server(
            name="structure",
            tools=[extract_structure]
        )

        self.options = ClaudeAgentOptions(
            cwd=project_path,
            allowed_tools=[
                "Read",
                "Write",
                "Glob",
                "mcp__structure__extract_structure"
            ],
            mcp_servers={"structure": structure_server},
            system_prompt="""You are a technical writer. Generate clear,
            comprehensive documentation following these guidelines:
            - Use clear, concise language
            - Include code examples
            - Document all parameters and return values
            - Add usage examples
            """,
            permission_mode="acceptEdits"
        )

    async def generate_docs_for_file(self, file_path: str):
        """Generate documentation for a single file."""
        async with ClaudeSDKClient(options=self.options) as client:
            await client.query(
                f"Generate comprehensive documentation for {file_path}. "
                f"Use the extract_structure tool to understand the code structure, "
                f"then create a markdown file with complete documentation."
            )

            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, TextBlock):
                            print(block.text)

    async def generate_api_reference(self):
        """Generate complete API reference documentation."""
        async with ClaudeSDKClient(options=self.options) as client:
            await client.query(
                "Generate a complete API reference document. "
                "1. Find all Python files in src/ "
                "2. Extract their structure using the extract_structure tool "
                "3. Read each file to understand the implementation "
                "4. Generate comprehensive API docs in docs/api.md"
            )

            async for msg in client.receive_response():
                pass

            print("API reference generated in docs/api.md")

    async def update_readme(self):
        """Update README with current project information."""
        async with ClaudeSDKClient(options=self.options) as client:
            await client.query(
                "Read the current README.md and analyze the codebase. "
                "Update the README to accurately reflect the current state, "
                "including new features, updated examples, and corrected descriptions."
            )

            async for msg in client.receive_response():
                pass

# Usage
async def main():
    docs = DocumentationGenerator("/path/to/project")

    # Generate docs for specific file
    await docs.generate_docs_for_file("src/main.py")

    # Generate complete API reference
    await docs.generate_api_reference()

    # Update README
    await docs.update_readme()

anyio.run(main)
```

---

## Additional Resources

- **SDK Documentation**: https://docs.anthropic.com/en/docs/claude-code/sdk/sdk-python
- **Claude Code Documentation**: https://docs.anthropic.com/en/docs/claude-code
- **Examples Directory**: `examples/` in this repository
- **Issue Tracker**: https://github.com/anthropics/claude-agent-sdk-python/issues
- **MCP Documentation**: https://modelcontextprotocol.io

## Contributing

Contributions are welcome! Please see the repository for contribution guidelines.

## License

MIT License - see LICENSE file for details.
