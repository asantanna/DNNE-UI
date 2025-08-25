# DNNE Development Status

*Last Updated: 2025-08-25*

## Latest Achievements (Current Week)

### 2025-08-25: Label Rats Nest Visualization
- ✅ Implemented rats nest feature for label connections
- Shows cyan lines connecting all labels in same network when selected
- Includes pulsing animation on selected labels
- Properly handles pan/zoom using LiteGraph's `toCanvasContext()`
- Performance optimized with label network caching

### 2025-08-22: Label System Enhancements
- ✅ Dictionary-free label system implementation
- ✅ Label node connection analyzer and repair tool
- ✅ Frontend patch verification integrated into startup

## Essential Commands

### Server Operations
```bash
dnne.bat                  # Start DNNE server (Windows)
./dnne_test              # Run test suite
./build_frontend.sh      # Build frontend
```

### Export System
```bash
python claude_scripts/programmatic_export.py  # Export workflows programmatically
```

### MCP Server (for Claude Desktop)
```bash
cd mcp_dnne_ui && python run_mcp_server.py
```

## Key Ports
- **8188**: DNNE UI Server
- **8585**: DNNE Agent Server  
- **8586**: Agent Client (remote)

## Key Documentation
- [CLAUDE.md](../CLAUDE.md) - Main project guidance
- [Task Index](for_claude/tasks/INDEX.md) - Current task status
- [Architecture](architecture/) - System design docs
- [Features](features/) - Feature documentation

## Recent Commits
```
670646c6 Complete dictionary-free label system implementation
9f185dd3 Fix analyzer to detect Label node connection issues
2f77b222 Enhance workflow analyzer with proper label verification and repair
459540b9 Integrate frontend patch verification into DNNE startup
4bba9e63 Implement patch system for npm packages and fix multi-connection rendering
```

## Active Development Focus
- UI Proxy implementation for server-initiated JavaScript
- Log viewer streaming completion fix
- Workflow naming and tab management fixes