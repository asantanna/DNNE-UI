# Documentation Consolidation Todo List

This document tracks the progress of consolidating and improving DNNE documentation.

## Goals
1. Reduce CLAUDE.md files to concise pointers (~80% size reduction)
2. Eliminate duplicate documentation
3. Create clear navigation structure
4. Move documentation to appropriate locations in docs-dnne/

## Progress Tracking

### 1. Delete Temporary Documentation
- [x] Delete `CONTEXT_MIGRATION.md` (temporary migration notes) - Completed 2025-07-28
- [x] Delete `claude_scripts/TEST_SUITE_CONSOLIDATION.md` (temporary consolidation notes) - Completed 2025-07-28
- [x] Delete `export_system/exports/balancing-test-settings.md` (test artifact) - Completed 2025-07-28
- [x] Delete `COMFYUI-README.md` (original ComfyUI readme, not relevant to DNNE) - Completed 2025-07-28

### 2. Shrink CLAUDE.md Files (Make them pointers, not documentation)

#### Main `/CLAUDE.md` (260 lines → ~50 lines)
- [x] Keep: Brief project overview (2-3 sentences) - Completed 2025-07-28
- [x] Keep: Repository paths and structure - Completed 2025-07-28
- [x] Keep: Essential commands only (conda activate, export, run) - Completed 2025-07-28
- [x] Remove: Detailed architecture sections - Completed 2025-07-28
- [x] Remove: Export system details - Completed 2025-07-28
- [x] Remove: Node implementation details - Completed 2025-07-28
- [x] Add: Pointers to docs-dnne for each topic - Completed 2025-07-28

#### `/export_system/CLAUDE.md` (413 lines → ~30 lines)
- [x] Keep: One-paragraph overview - Completed 2025-07-28
- [x] Keep: Directory structure - Completed 2025-07-28
- [x] Remove: All detailed explanations - Completed 2025-07-28
- [x] Add: "See docs-dnne/architecture/export_system.md" - Completed 2025-07-28

#### `/export_system/templates/CLAUDE.md` (692 lines → ~30 lines)
- [x] Keep: Brief description - Completed 2025-07-28
- [x] Keep: File naming conventions - Completed 2025-07-28
- [x] Remove: All template examples and patterns - Completed 2025-07-28
- [x] Create: `docs-dnne/architecture/templates.md` with the removed content - Completed 2025-07-28

#### `/custom_nodes/ml_nodes/CLAUDE.md` (545 lines → ~30 lines)
- [x] Keep: List of node categories - Completed 2025-07-28
- [x] Remove: Implementation details - Completed 2025-07-28
- [x] Add: "See docs-dnne/nodes/ml/" - Completed 2025-07-28

#### `/custom_nodes/robotics_nodes/CLAUDE.md` (347 lines → ~30 lines)
- [x] Keep: List of node types - Completed 2025-07-28
- [x] Remove: Implementation details - Completed 2025-07-28
- [x] Add: "See docs-dnne/nodes/robotics/" - Completed 2025-07-28

### 3. Improve Root Documentation (Keep at root but enhance)

#### `README.md` (root)
- [ ] Add clear project overview
- [ ] Add quick start section
- [ ] Add navigation to docs-dnne
- [ ] Keep installation/setup basics

#### `CONFIGURATION_GUIDE.md` (root)
- [ ] Review and update for accuracy
- [ ] Add more examples
- [ ] Ensure all paths are current

### 4. Consolidate Duplicate/Scattered Content

#### Adaptive Yielding Documentation
- [ ] Keep main: `docs-dnne/architecture/adaptive-yielding.md`
- [ ] Archive: Move `docs-dnne/experiments/yield_tests/` to `experiments/archive/`
- [ ] Merge any unique content from experiments into main doc

#### Architecture Documentation
- [ ] Check overlap between `system-balancing.md` and `adaptive-yielding.md`
- [ ] Check overlap between `async-environment-design.md` and `queue_framework.md`
- [ ] Merge duplicated content

#### Code Review Documentation
- [ ] Move content from `DNNE_CODE_REVIEW.md` to `docs-dnne/development/code-review.md`
- [ ] Delete original file

### 5. Create Missing Documentation
- [ ] Create `docs-dnne/architecture/templates.md` (from templates/CLAUDE.md content)
- [ ] Create `docs-dnne/setup/environment.md` (conda and Isaac Gym setup from main CLAUDE.md)
- [ ] Update `docs-dnne/README.md` with better navigation

### 6. Review and Update .claude Directory
- [ ] Keep `.claude/commands/dev-status.md` but remove duplicate content about adaptive yielding
- [ ] Review `.claude/agents/dnne-code-reviewer.md` for relevance

## Completion Criteria
- All CLAUDE.md files < 50 lines
- No duplicate documentation across files
- Clear pointers from CLAUDE.md to actual docs
- All documentation in logical locations
- Navigation is intuitive

## Notes
- When marking items complete, use [x] instead of [ ]
- Add completion date and any relevant notes
- If content is moved, note the destination