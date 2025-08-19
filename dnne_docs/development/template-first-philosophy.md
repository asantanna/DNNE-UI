# Template-First Development Philosophy

## Core Principle

**"Fix templates and re-export. Haven't you learned the lesson, yet?"**

This fundamental principle governs all DNNE development: Templates are the source of truth, and generated code is ephemeral. NEVER modify generated code directly - always fix the templates and re-export.

## Why Template-First?

### 1. **Single Source of Truth**
Templates define the canonical implementation. When you fix a bug in a template, it's fixed for all future exports. When you hack generated code, the bug returns on the next export.

### 2. **Consistency Across Exports**
Every workflow that uses a node type gets the same, tested implementation. Hacking individual exports creates inconsistency and maintenance nightmares.

### 3. **Version Control Clarity**
Templates are version-controlled and reviewed. Generated code is transient and can be regenerated at any time. This keeps the repository clean and changes traceable.

### 4. **Prevents Regression**
Fixes in templates persist. Fixes in generated code vanish on re-export, causing "fixed" bugs to mysteriously reappear.

## The Template-First Workflow

### ✅ Correct Approach

1. **Identify Issue**: Find bug in exported workflow
2. **Trace to Template**: Locate the template that generated the buggy code
3. **Fix Template**: Update the template file
4. **Re-export**: Generate fresh code from fixed template
5. **Test**: Verify the fix works
6. **Commit Template**: Only templates go into version control

```bash
# Example: Fixing initialization order issue
# 1. Find the template
vi export_system/templates/framework/runner.tpl

# 2. Add initialization call before node creation
# Line 344: g.init_system_ready()

# 3. Re-export the workflow
python claude_scripts/programmatic_export.py --workflow Franka_Coop_Nodes

# 4. Test the fixed export
cd export_system/exports/Franka_Coop_Nodes
python runner.py --timeout 15

# 5. Commit the template fix
git add export_system/templates/framework/runner.tpl
git commit -m "Fix initialization order in runner template"
```

### ❌ Wrong Approach (Never Do This!)

```bash
# BAD: Editing generated code directly
vi export_system/exports/Franka_Coop_Nodes/runner.py
# Adding g.init_system_ready() directly to exported file
# This "fix" will be lost on next export!
```

## Template vs Generated Code Boundaries

### Templates (`export_system/templates/`)
- **Permanent**: Checked into version control
- **Canonical**: Define the true implementation
- **Maintained**: Updated with bug fixes and features
- **Reviewed**: Subject to code review
- **Documented**: Include comments and documentation

### Generated Code (`export_system/exports/`)
- **Ephemeral**: Can be deleted and regenerated
- **Disposable**: Never manually edited
- **Gitignored**: Not tracked in version control (usually)
- **Testing Only**: Used for testing and debugging
- **Read-Only**: Treat as read-only output

## Common Scenarios

### Scenario 1: Node Behavior Bug
**Problem**: ConcatNode crashes on 1D tensors
**Wrong**: Edit `exports/MyWorkflow/nodes/concatnode_42.py`
**Right**: Fix `templates/nodes/concat_node_queue.tpl`

### Scenario 2: Missing Initialization
**Problem**: Nodes start before connections established
**Wrong**: Add init call to `exports/MyWorkflow/runner.py`
**Right**: Update `templates/framework/runner.tpl`

### Scenario 3: Import Order Issue
**Problem**: Isaac Gym must import before PyTorch
**Wrong**: Reorder imports in exported runner.py
**Right**: Fix import order in `templates/framework/runner.tpl`

## Template Development Guidelines

### 1. **Make Templates Self-Contained**
Templates should include all necessary imports, error handling, and documentation.

### 2. **Use Clear Variable Names**
Template variables should be descriptive:
```python
# Good
{NODE_ID}
{LEARNING_RATE}
{CHECKPOINT_DIR}

# Bad
{VAR1}
{PARAM}
{DIR}
```

### 3. **Document Template Variables**
Always document what variables the template expects:
```python
# Template variables - replaced during export
# NODE_ID: Unique node identifier (e.g., "42")
# CLASS_NAME: Node class name (e.g., "NetworkNode")
# HIDDEN_SIZE: Network hidden layer size
# DEVICE: Computation device ("cuda" or "cpu")
```

### 4. **Include Fail-Fast Validation**
Templates should validate their inputs:
```python
if not {REQUIRED_PARAM}:
    raise ValueError(f"Node {NODE_ID} missing required parameter REQUIRED_PARAM")
```

### 5. **Version Template Changes**
Use meaningful commit messages when updating templates:
```bash
git commit -m "Fix concat_node_queue.tpl to handle 1D tensor inputs

- Add automatic unsqueeze for 1D tensors
- Maintain batch/feature dimension convention
- Add dimension validation with clear errors"
```

## Testing Template Changes

### 1. **Test Multiple Workflows**
A template change affects ALL workflows using that node type:
```bash
# Test all affected workflows
python claude_scripts/test_all_exports.py
```

### 2. **Verify Variable Substitution**
Check that all template variables are replaced:
```bash
# Bad: Variables not replaced
grep -r "{.*}" export_system/exports/MyWorkflow/
```

### 3. **Check Edge Cases**
Templates must handle various configurations:
- Missing optional parameters
- Different data types
- Various connection patterns

## Anti-Patterns to Avoid

### ❌ **The Quick Fix**
"I'll just fix this one export now and update the template later"
- Later never comes
- Next export loses your fix
- Other workflows don't benefit

### ❌ **The Special Case**
"This workflow needs something different, so I'll hack the export"
- Makes workflow unmaintainable
- Can't regenerate without losing changes
- Better: Add configuration option to template

### ❌ **The Debug Edit**
"I'll add some debug prints to the exported code"
- Use proper logging in templates instead
- Add --debug flag support
- Debug code should be reusable

### ❌ **The Monkey Patch**
"I'll just patch this one method in the export"
- Template architecture may change
- Patches break on regeneration
- Fix the root cause in templates

## Emergency Procedures

If you absolutely MUST modify generated code temporarily:

1. **Document it prominently**:
```python
# TEMPORARY HACK - DO NOT COMMIT
# TODO: Fix in template concat_node_queue.tpl
# Issue: Handles 1D tensors incorrectly
```

2. **Create a template fix task immediately**
3. **Never commit generated code changes**
4. **Test the template fix ASAP**

## Cultural Reinforcement

The template-first philosophy is not just a technical practice—it's a cultural value in DNNE development:

- **Code reviews** should reject any PR with modified generated code
- **Documentation** should emphasize templates as source of truth
- **Testing** should validate templates, not specific exports
- **Debugging** should trace issues back to templates
- **Training** for new developers should emphasize this principle

## Remember

> "No more hacking exported code. Fix templates and re-export."

This is not a suggestion—it's a fundamental requirement for maintaining DNNE's code quality, consistency, and maintainability. Every shortcut taken by modifying generated code creates technical debt that compounds over time.

When in doubt, ask yourself: "Am I fixing the template or hacking the export?" If it's the latter, stop and fix the template.