# Documentation for Claude Sessions

This directory contains key documentation to help Claude understand and work with the DNNE codebase effectively.

## Current Documentation

### DNNE server-agent and client-agent
- **`dnne_agent/TASKS.md`** - Status and task list for the agent server and agent client.
- **`docs-dnne/architecture/dnne-agent.md`** - DNNE agent architecture and message interactions.

### Performance Documentation
- **`perf_analysis/performance_analysis_overview.md`** - Current performance analysis and optimization opportunities

## Key Principles for Claude

1. **Don't guess - instrument and compare**: Always add debug prints to understand behavior
2. **Match IGE patterns**: DNNE should inherit from IGE with minimal changes
3. **Test incrementally**: Fix one issue at a time and verify
4. **Document discoveries**: Update guides with new insights

Last updated: January 2025