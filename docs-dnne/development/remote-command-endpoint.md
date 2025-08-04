# Remote Command Endpoint for DNNE Server

## Overview

This proposal outlines the addition of a remote command endpoint to the DNNE server, enabling Claude Code and other tools to programmatically control and manage the server. This capability is essential for automated testing, debugging, and server lifecycle management.

## Motivation

Currently, Claude Code cannot:
- Restart the DNNE server after making configuration changes
- Reload custom nodes after modifications
- Clear caches or reset server state
- Retrieve server logs or diagnostic information
- Trigger specific test scenarios

A remote command endpoint would provide these capabilities in a secure, structured way.

## Proposed API Design

### Endpoint
```
POST /api/remote_command
```

### Request Structure
```json
{
  "command": "string",
  "args": {
    // Command-specific arguments
  },
  "auth": "optional_auth_token",
  "request_id": "optional_unique_id"
}
```

### Response Structure
```json
{
  "success": boolean,
  "command": "string",
  "message": "string",
  "data": {
    // Command-specific response data
  },
  "request_id": "optional_unique_id",
  "timestamp": "ISO-8601 timestamp"
}
```

## Core Commands

### 1. Server Management

#### restart
Restart the DNNE server process
```json
{
  "command": "restart",
  "args": {
    "delay": 2,        // Seconds before restart (default: 2)
    "reason": "string", // Optional restart reason for logging
    "preserve_args": true // Keep current command line arguments
  }
}
```

#### shutdown
Gracefully shutdown the server
```json
{
  "command": "shutdown",
  "args": {
    "delay": 5,
    "save_state": true
  }
}
```

#### get_status
Get server status and health information
```json
{
  "command": "get_status",
  "args": {}
}
// Response data:
{
  "uptime": 3600,
  "version": "1.0.0",
  "memory_usage": 512000000,
  "active_workflows": 2,
  "node_count": 45,
  "agent_connected": true
}
```

### 2. Node Management

#### reload_nodes
Reload custom nodes without restarting
```json
{
  "command": "reload_nodes",
  "args": {
    "node_path": "optional/specific/path",
    "force": false  // Force reload even if errors
  }
}
```

#### list_nodes
Get list of available nodes
```json
{
  "command": "list_nodes",
  "args": {
    "category": "ml_nodes",  // Optional filter
    "include_metadata": true
  }
}
```

#### validate_node
Validate a specific node implementation
```json
{
  "command": "validate_node",
  "args": {
    "node_type": "LinearLayer",
    "check_exports": true
  }
}
```

### 3. Cache Management

#### clear_cache
Clear various server caches
```json
{
  "command": "clear_cache",
  "args": {
    "type": "all" | "models" | "nodes" | "workflows",
    "force": false
  }
}
```

#### get_cache_info
Get cache statistics
```json
{
  "command": "get_cache_info",
  "args": {}
}
```

### 4. Workflow Management

#### list_workflows
Get available workflows
```json
{
  "command": "list_workflows",
  "args": {
    "path": "user/default/workflows"
  }
}
```

#### validate_workflow
Validate a workflow JSON
```json
{
  "command": "validate_workflow",
  "args": {
    "workflow_name": "MNIST_Test",
    "check_nodes": true,
    "check_connections": true
  }
}
```

#### export_workflow
Trigger workflow export programmatically
```json
{
  "command": "export_workflow",
  "args": {
    "workflow_name": "MNIST_Test",
    "target": "local" | "client_id",
    "run_after": false
  }
}
```

### 5. Debugging and Diagnostics

#### get_logs
Retrieve recent server logs
```json
{
  "command": "get_logs",
  "args": {
    "lines": 100,
    "level": "ERROR" | "WARNING" | "INFO" | "DEBUG",
    "since": "timestamp"
  }
}
```

#### set_log_level
Change logging level dynamically
```json
{
  "command": "set_log_level",
  "args": {
    "level": "DEBUG",
    "subsystem": "export_system"  // Optional
  }
}
```

#### get_errors
Get recent errors and exceptions
```json
{
  "command": "get_errors",
  "args": {
    "limit": 10,
    "include_stack": true
  }
}
```

### 6. Testing Support

#### run_test
Execute specific test scenarios
```json
{
  "command": "run_test",
  "args": {
    "test_name": "export_mnist",
    "timeout": 30
  }
}
```

#### mock_data
Inject mock data for testing
```json
{
  "command": "mock_data",
  "args": {
    "type": "workflow",
    "data": {}
  }
}
```

## Implementation Details

### Server-side Implementation (server.py)

```python
@routes.post("/api/remote_command")
async def handle_remote_command(request):
    """Handle remote command requests."""
    try:
        json_data = await request.json()
        
        # Optional authentication
        if self.require_auth:
            auth_token = json_data.get("auth")
            if not self.validate_auth(auth_token):
                return web.json_response({
                    "success": False,
                    "message": "Authentication failed"
                }, status=401)
        
        command = json_data.get("command")
        args = json_data.get("args", {})
        request_id = json_data.get("request_id")
        
        # Command handler dispatch
        handler = self.command_handlers.get(command)
        if not handler:
            return web.json_response({
                "success": False,
                "command": command,
                "message": f"Unknown command: {command}",
                "request_id": request_id
            }, status=400)
        
        # Execute command
        result = await handler(args)
        
        return web.json_response({
            "success": True,
            "command": command,
            "message": result.get("message", "Command executed successfully"),
            "data": result.get("data", {}),
            "request_id": request_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Remote command error: {e}")
        return web.json_response({
            "success": False,
            "message": str(e),
            "request_id": json_data.get("request_id")
        }, status=500)
```

### Restart Implementation

```python
async def handle_restart(args):
    """Handle server restart command."""
    delay = args.get("delay", 2)
    reason = args.get("reason", "Remote command")
    preserve_args = args.get("preserve_args", True)
    
    logging.info(f"Server restart requested: {reason}")
    
    # Schedule restart
    async def do_restart():
        await asyncio.sleep(delay)
        
        # Save current state if needed
        self.save_server_state()
        
        # Restart process
        if preserve_args:
            os.execv(sys.executable, [sys.executable] + sys.argv)
        else:
            os.execv(sys.executable, [sys.executable, "main.py"])
    
    asyncio.create_task(do_restart())
    
    return {
        "message": f"Server will restart in {delay} seconds",
        "data": {"delay": delay, "reason": reason}
    }
```

## Security Considerations

### Authentication Options

1. **Token-based**: Simple shared secret
   ```python
   auth_token = os.environ.get("DNNE_REMOTE_AUTH")
   ```

2. **IP Whitelist**: Only allow from localhost/specific IPs
   ```python
   allowed_ips = ["127.0.0.1", "::1"]
   ```

3. **Command Restrictions**: Limit available commands based on auth level
   ```python
   public_commands = ["get_status", "list_workflows"]
   admin_commands = ["restart", "shutdown", "clear_cache"]
   ```

### Rate Limiting
Prevent abuse with rate limiting:
```python
command_rate_limit = {
    "restart": (1, 300),  # 1 per 5 minutes
    "clear_cache": (5, 60),  # 5 per minute
    "get_logs": (10, 60)  # 10 per minute
}
```

### Audit Logging
Log all remote commands for security audit:
```python
logging.info(f"Remote command: {command} from {request.remote} with args: {args}")
```

## Configuration

Add to `dnne_config.json`:
```json
{
  "dnne": {
    "remote_command": {
      "enabled": true,
      "require_auth": true,
      "auth_token": "env:DNNE_REMOTE_AUTH",
      "allowed_ips": ["127.0.0.1", "::1"],
      "rate_limits": {
        "restart": [1, 300],
        "clear_cache": [5, 60]
      },
      "disabled_commands": [],
      "audit_log": true
    }
  }
}
```

## Testing

### Unit Tests
```python
async def test_restart_command():
    """Test server restart command."""
    response = await client.post("/api/remote_command", json={
        "command": "restart",
        "args": {"delay": 0, "reason": "test"}
    })
    assert response.status == 200
    data = await response.json()
    assert data["success"] == True
```

### Integration Tests
- Test command execution flow
- Verify authentication
- Check rate limiting
- Validate error handling

## Future Enhancements

1. **WebSocket Commands**: Real-time command execution
2. **Batch Commands**: Execute multiple commands in sequence
3. **Scheduled Commands**: Cron-like scheduling
4. **Command Plugins**: Extensible command system
5. **GUI Integration**: Add UI controls for common commands
6. **Command History**: Track and replay commands
7. **Conditional Commands**: Execute based on server state

## Benefits for Claude Code

With this endpoint, Claude Code can:
1. **Restart server** after modifying configuration files
2. **Reload nodes** after editing custom node code
3. **Clear caches** when debugging issues
4. **Retrieve logs** for error analysis
5. **Validate workflows** before export
6. **Run tests** programmatically
7. **Monitor server health** during operations

## Implementation Priority

1. **High Priority** (Week 1)
   - restart
   - get_status
   - get_logs

2. **Medium Priority** (Week 2)
   - reload_nodes
   - clear_cache
   - validate_workflow

3. **Low Priority** (Week 3+)
   - Advanced debugging commands
   - Test automation
   - Batch operations

## Success Criteria

1. **Functionality**: All core commands working reliably
2. **Security**: Proper authentication and rate limiting
3. **Performance**: <100ms response time for most commands
4. **Reliability**: Graceful error handling
5. **Documentation**: Complete API documentation

## Next Steps

1. Review and approve proposal
2. Implement core commands (restart, status, logs)
3. Add authentication and security
4. Create test suite
5. Document API usage
6. Integrate with Claude Code workflow