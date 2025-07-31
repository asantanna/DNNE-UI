# Documentation for Claude Sessions

This directory contains key documentation to help Claude understand and work with the DNNE codebase effectively.

## Current Documentation

### Core Guides
- **`dnne_debugging_guide.md`** - Comprehensive guide on debugging DNNE to match IsaacGymEnvs behavior. Contains lessons learned, patterns, and techniques from successfully fixing PPO implementation.

### Performance Documentation
- **`performance_analysis_overview.md`** - Current performance analysis and optimization opportunities
- **`performance_work_log.md`** - Historical log of performance optimization attempts

### Archived Documentation
The `archived_debug_docs/` directory contains older documentation that has been superseded by the consolidated debugging guide. These files are kept for historical reference but should not be used as primary guidance.

## When to Use Each Document

1. **Starting a debugging session**: Read `dnne_debugging_guide.md` first
2. **Working on performance**: Review both performance documents
3. **Understanding history**: Check archived docs only if needed for context

## Key Principles for Claude

1. **Don't guess - instrument and compare**: Always add debug prints to understand behavior
2. **Match IGE patterns**: DNNE should inherit from IGE with minimal changes
3. **Test incrementally**: Fix one issue at a time and verify
4. **Document discoveries**: Update guides with new insights

Last updated: January 2025