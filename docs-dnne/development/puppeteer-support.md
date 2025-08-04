# Enhanced Puppeteer Support for Claude Code

## Overview

This proposal outlines enhancements to the MCP Puppeteer server to provide Claude Code with advanced browser debugging capabilities for DNNE development. These capabilities would enable real-time debugging, performance profiling, and comprehensive testing of the DNNE UI.

## Current State

### Available MCP Puppeteer Functions
- `puppeteer_navigate`: Navigate to URLs
- `puppeteer_screenshot`: Capture screenshots
- `puppeteer_click`: Click elements
- `puppeteer_fill`: Fill input fields
- `puppeteer_select`: Select dropdown options
- `puppeteer_hover`: Hover over elements
- `puppeteer_evaluate`: Execute JavaScript in page context

### Limitations
- No access to console logs (console.log, console.error, etc.)
- No Chrome DevTools Protocol (CDP) access
- Cannot set breakpoints or pause execution
- No network request/response inspection
- No performance profiling capabilities
- No access to browser errors or warnings

## Proposed Enhancements

### Phase 1: Console and Error Capture

#### New Functions
```typescript
puppeteer_get_console_logs(options?: {
  level?: 'log' | 'warn' | 'error' | 'debug' | 'all',
  since?: timestamp,
  limit?: number
}): ConsoleMessage[]

puppeteer_get_page_errors(): PageError[]

puppeteer_clear_logs(): void
```

#### Implementation
- Attach listeners to page console events
- Buffer messages with configurable retention
- Include stack traces and source locations
- Capture uncaught exceptions and promise rejections

### Phase 2: Chrome DevTools Protocol Access

#### New Functions
```typescript
puppeteer_cdp_send(method: string, params?: object): any

puppeteer_enable_domain(domain: 'Debugger' | 'Runtime' | 'Network' | 'Performance'): void

puppeteer_on_cdp_event(event: string, callback: Function): void
```

#### Key CDP Domains
1. **Debugger**: Breakpoints, stepping, call frames
2. **Runtime**: Expression evaluation, object inspection
3. **Network**: Request interception, response monitoring
4. **Performance**: Metrics, memory profiling

### Phase 3: Advanced Debugging Features

#### Breakpoint Management
```typescript
puppeteer_set_breakpoint(options: {
  url?: string,
  urlRegex?: string,
  lineNumber: number,
  columnNumber?: number
}): BreakpointId

puppeteer_remove_breakpoint(breakpointId: string): void

puppeteer_pause(): void
puppeteer_resume(): void
puppeteer_step_over(): void
puppeteer_step_into(): void
```

#### State Inspection
```typescript
puppeteer_get_call_frames(): CallFrame[]

puppeteer_evaluate_on_call_frame(
  callFrameId: string,
  expression: string
): EvaluationResult

puppeteer_get_scope_variables(scopeId: string): Variable[]
```

#### Network Monitoring
```typescript
puppeteer_get_network_logs(options?: {
  includeData?: boolean,
  filter?: RequestFilter
}): NetworkEntry[]

puppeteer_intercept_request(
  pattern: string,
  handler: (request: Request) => void
): void
```

## DNNE-Specific Use Cases

### 1. Vue Component Debugging
- Set breakpoints in component lifecycle methods
- Inspect component data and computed properties
- Trace prop changes and event emissions
- Debug Vuex state mutations

### 2. WebSocket Debugging
```javascript
// Intercept and log WebSocket messages
await puppeteer_evaluate(`
  const originalSend = WebSocket.prototype.send;
  WebSocket.prototype.send = function(data) {
    console.log('[WS Send]', data);
    return originalSend.call(this, data);
  };
`);
```

### 3. Node Execution Tracing
- Monitor node creation and deletion
- Track data flow between nodes
- Debug connection issues
- Profile node execution performance

### 4. Export System Validation
- Verify export button states
- Confirm file generation
- Validate error handling
- Test agent client selection

### 5. Performance Analysis
- Identify rendering bottlenecks
- Memory leak detection
- Network request optimization
- JavaScript execution profiling

## Implementation Approach

### Option 1: Extend MCP Puppeteer Server
**Pros**: 
- Leverages existing infrastructure
- Maintains compatibility
- Incremental rollout possible

**Cons**: 
- Requires upstream changes
- May have limitations

**Implementation Steps**:
1. Fork MCP Puppeteer server
2. Add console capture capabilities
3. Expose CDP session access
4. Submit PR or maintain fork

### Option 2: Create DNNE Browser Automation Tool
**Pros**: 
- Full control over features
- DNNE-specific optimizations
- No external dependencies

**Cons**: 
- More development effort
- Separate maintenance

**Architecture**:
```
Claude Code <-> DNNE Debug Server <-> Chrome/Chromium
                       |
                  CDP Session
```

## Immediate Workarounds

Until full implementation, Claude Code can use:

### 1. Debug Helper Injection
```javascript
await puppeteer_evaluate(`
  window.DNNE_Debug = {
    logs: [],
    breakpoints: {},
    
    captureConsole() {
      ['log', 'warn', 'error'].forEach(method => {
        const original = console[method];
        console[method] = (...args) => {
          this.logs.push({
            type: method,
            args: args,
            timestamp: Date.now(),
            stack: new Error().stack
          });
          original.apply(console, args);
        };
      });
    },
    
    setBreakpoint(label, condition) {
      this.breakpoints[label] = condition;
      // Use debugger statement when condition met
      if (condition()) {
        debugger;
      }
    },
    
    getState() {
      return {
        logs: this.logs,
        errors: window.errors || [],
        performance: performance.getEntriesByType('measure')
      };
    }
  };
  
  DNNE_Debug.captureConsole();
`);
```

### 2. Periodic State Polling
```javascript
// Poll for debug information every second
setInterval(async () => {
  const debugInfo = await puppeteer_evaluate('DNNE_Debug.getState()');
  // Process debug information
}, 1000);
```

## Testing Strategy

### Unit Tests
- Console capture accuracy
- Error handling robustness
- CDP message formatting

### Integration Tests
- Full debugging session flow
- Breakpoint hit verification
- Network interception

### DNNE-Specific Tests
- Vue component debugging
- WebSocket message capture
- Export workflow validation

## Security Considerations

1. **Sandbox Isolation**: Ensure debugger access doesn't break sandbox
2. **Data Filtering**: Sanitize sensitive information in logs
3. **Access Control**: Limit CDP access to safe domains
4. **Resource Limits**: Cap log buffer sizes

## Timeline Estimate

- **Phase 1** (Console/Errors): 1 week
- **Phase 2** (CDP Access): 2 weeks  
- **Phase 3** (Full Debugging): 2 weeks
- **Testing & Documentation**: 1 week

Total: ~6 weeks for full implementation

## Success Metrics

1. **Functionality**: All proposed features working
2. **Performance**: <100ms latency for debug operations
3. **Reliability**: 99% success rate for debug commands
4. **Coverage**: Can debug all DNNE UI components

## References

- [Chrome DevTools Protocol](https://chromedevtools.github.io/devtools-protocol/)
- [Puppeteer API](https://pptr.dev/api)
- [MCP Specification](https://modelcontextprotocol.io/docs)
- [Vue.js DevTools Protocol](https://github.com/vuejs/devtools)

## Next Steps

1. Review and approve proposal
2. Decide on implementation approach
3. Create development branch
4. Begin Phase 1 implementation
5. Iterate based on testing feedback