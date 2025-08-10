# Documentation for Claude Sessions

This directory contains key documentation and task tracking to help Claude understand and work with the DNNE codebase effectively.

## Directory Structure

### Task Tracking (`tasks/`)
The `tasks/` subdirectory contains component-specific task tracking files for ongoing development:

- **`tasks/INDEX.md`** - Quick overview of all component task statuses
- **`tasks/MCP/TASKS.md`** - MCP (Model Context Protocol) browser automation tasks
- **`tasks/dnne_agent/TASKS.md`** - DNNE Agent integration for remote deployment
- **`tasks/log_window/TASKS.md`** - Log window UI implementation tasks
- **`tasks/TEMPLATE.md`** - Template for creating new task tracking files

### Key Documentation References

#### Architecture & Design
- **`dnne-docs/architecture`** - Various documents describing the architecture and special issues in DNNE

#### Quick Access
- **Task Status**: Check `tasks/INDEX.md` for current priorities and progress
- **New Tasks**: Use `tasks/TEMPLATE.md` when creating new task files

## Guidelines for Claude Code

### Task Management
1. **Check INDEX.md first** - Get quick overview of all component statuses
2. **Update task files** - Mark tasks complete as you work
3. **Document blockers** - Clearly note dependencies and issues
4. **Use the template** - Create consistent task files for new components

### Key Principles
1. **Don't guess - instrument and compare**: Always add debug prints to understand behavior
2. **Test incrementally**: Fix one issue at a time and verify
3. **Document discoveries**: Update guides with new insights
4. **Track progress**: Use task files to maintain continuity between sessions

## Current Priorities

Based on task statuses (see `tasks/INDEX.md` for details):
1. MCP log management functions
2. Log window UI implementation
3. Documentation updates

## Quick Commands

```bash
# Check all task statuses
cat dnne-docs/for_claude/tasks/INDEX.md

# View specific component tasks
cat dnne-docs/for_claude/tasks/{component}/TASKS.md

# Create new task file from template
cp dnne-docs/for_claude/tasks/TEMPLATE.md dnne-docs/for_claude/tasks/{new_component}/TASKS.md
```

Last updated: 2025-08-07