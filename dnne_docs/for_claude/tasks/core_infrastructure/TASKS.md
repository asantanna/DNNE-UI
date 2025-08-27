# Core Infrastructure - Task Tracking

*For historical accomplishments, see HISTORY.md*

## Current Status
**Complete** - All features operational (100%)
- Core server fully functional
- ✅ Windows URL display fixed (shows localhost instead of 0.0.0.0)

## 📋 TODO

### Logging System Cleanup (Low Priority)

1. **Change --verbose and --debug to --log-level INFO|DEBUG in runner argument parser**
   - Replace confusing dual flags with single clear option
   - Default to INFO level for better user experience
   
2. **Update configure_logging to use simplified log-level approach**
   - Simplify the logic to just set global log level
   - Remove complex subsystem parsing from level setting
   
3. **Add --heartbeat flag separate from debug mode**
   - Decouple monitoring feature from log level
   - Enable heartbeat with `--heartbeat` regardless of log level
   
4. **Make --debug continue to enable heartbeat for backwards compatibility**
   - Ensure `--debug` still enables heartbeat as before
   - Heartbeat check becomes: `if g.debug or g.heartbeat:`
   
5. **Update runner_args.json to match new argument structure**
   - Ensure UI configuration stays in sync with CLI changes
   - Add UI controls for new --heartbeat flag
   
6. **Test new logging configuration with various workflows**
   - Verify changes work with MNIST, Franka, and other workflows
   - Ensure backwards compatibility is maintained
   
7. **Rename any DEBUG subsystem to avoid confusion with log level**
   - Find and rename any subsystems called "DEBUG" to something descriptive
   - Prevents confusion between subsystem name and log level

### Notes on Current Issues:
- `--verbose` accepts subsystems but also treats "DEBUG" as a subsystem name (confusing!)
- `--debug` controls both log level AND the heartbeat feature (mixed concerns)
- Default WARNING level hides too much from users
- Heartbeat should be an independent monitoring feature

## 💡 Quick Reference

### Important Files
- `server.py` - Main server implementation
- `main.py` - Server initialization and startup
- `dnne_config.json` - Configuration settings
- `dnne_hooks/` - DNNE-specific server extensions

### Testing Commands
```bash
# Start server (Windows)
./dnne.bat

# Check server health
curl http://localhost:8188/health

# Remote command test
python claude_scripts/test_remote_command.py
```

## Future Enhancements
1. Add server metrics/monitoring dashboard
2. Implement server-side workflow validation
3. Add workflow versioning system
4. Improve error messages and logging
5. Add server-side backup/restore functionality