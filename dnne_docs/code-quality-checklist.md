# DNNE Code Quality Checklist

This document contains critical rules and best practices that must be followed when developing DNNE code. Use this checklist regularly to ensure code quality and prevent recurring bugs.

## 🚨 Critical Rules (Must Never Violate)

### 1. **Fail-Fast Paradigm**
- ❌ **NEVER use fallback values when data is missing**
- ✅ **ALWAYS throw clear errors with specific diagnostic information**
- ❌ **NEVER guess default values that hide real problems**

**Bad Example:**
```python
# This hides bugs and wastes debugging time
num_envs = widget_values[1] if len(widget_values) > 1 else 16
```

**Good Example:**
```python
# This reveals problems immediately
if len(widget_values) < 2:
    raise ValueError(f"Node {node_id} missing num_envs parameter. Got {len(widget_values)} values, expected at least 2")
num_envs = widget_values[1]
```

### 2. **Universal Parameter Reading**
- ✅ **ALWAYS use `cls.get_node_parameters_batch()` for widget values**
- ❌ **NEVER directly access `widgets_values` or `inputs`**
- ✅ **Define parameter specifications with proper names and types**

**Correct Pattern:**
```python
@classmethod
def prepare_template_vars(cls, node_id, node_data, connections, node_registry=None, all_nodes=None, all_links=None):
    # Use universal parameter reader for consistent data access
    param_specs = [
        {'name': 'learning_rate', 'default': 0.001},
        {'name': 'batch_size', 'default': 32},
        {'name': 'device', 'default': 'cuda'}
    ]
    
    params = cls.get_node_parameters_batch(node_data, param_specs)
    
    return {
        "LEARNING_RATE": params['learning_rate'],
        "BATCH_SIZE": params['batch_size'],
        "DEVICE": params['device']
    }
```

### 3. **Base Class Implementation**
- ✅ **ALWAYS throw `NotImplementedError` when subclasses must implement methods**
- ❌ **NEVER implement "guessed" default behavior in base classes**
- ✅ **Use clear error messages explaining what needs to be implemented**

**Correct Pattern:**
```python
class BaseNode:
    def get_input_names(self):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement get_input_names() method")
```

### 4. **Test Completion Standards**
- ✅ **ONLY mark tests as complete when they execute successfully from start to finish**
- ❌ **NEVER mark tests complete if export fails, code crashes, or functionality is missing**
- ✅ **Document failures honestly - partial success is not success**

**Test Status Rules:**
- **COMPLETE**: Test runs successfully without errors
- **FAILED**: Test crashes, throws exceptions, or produces wrong results  
- **INCOMPLETE**: Export fails, dependencies missing, or setup issues
- **PENDING**: Test exists but functionality not yet implemented

## 📁 File Organization Rules

### 5. **File Creation Restrictions**
- ❌ **ABSOLUTE PROHIBITION: NEVER create ANY files in project root** (`/mnt/e/ALS-Projects/DNNE/DNNE-UI/`)
- ✅ **Exports MUST go to**: `export_system/exports/{workflow_name}/` ONLY
- ✅ **Test files MUST go to**: `claude_scripts/` or `tests-dnne/` ONLY
- ✅ **ALWAYS double-check export paths before running ANY export command**

### 6. **Template Organization**
- ✅ **Queue templates end with `_queue.py`** for async execution
- ✅ **Node exporters mirror the `custom_nodes/` directory structure**
- ✅ **Generated code follows consistent naming patterns**

## 🔧 Development Patterns

### 7. **Isaac Gym Import Order**
- ✅ **CRITICAL: ALWAYS import `isaacgym` before `torch`**
- ✅ **The export system MUST ensure Isaac Gym nodes are imported first**
- ❌ **PyTorch imported before Isaac Gym causes "PyTorch was imported before isaacgym" errors**

**Correct Pattern:**
```python
# Always import Isaac Gym first
import isaacgym
from isaacgym import gymapi, gymtorch
# Then import torch
import torch
```

### 8. **Error Handling in Templates**
- ✅ **Include proper error handling in all node compute methods**
- ✅ **Log errors with context information**
- ✅ **Provide meaningful error messages for debugging**

**Template Pattern:**
```python
async def compute(self, input_data):
    try:
        # Main computation
        result = self.process_data(input_data)
        return {"output": result}
        
    except Exception as e:
        self.logger.error(f"Error in {self.node_id}: {e}")
        # Either return safe default or re-raise with context
        raise RuntimeError(f"Node {self.node_id} computation failed: {e}")
```

### 9. **Template Variable Substitution**
- ✅ **Use descriptive variable names**: `LEARNING_RATE` not `LR`
- ✅ **Provide sensible defaults in template_vars**
- ✅ **Use correct quoting**: `"{STRING_VAR}"` vs `{NUMERIC_VAR}`
- ✅ **Group related variables logically**

### 10. **Export System Integration**
- ✅ **Every new node type requires template + exporter + registration**
- ✅ **Templates must use queue-based async patterns**
- ✅ **Exporters must validate input parameters**
- ✅ **Registration must be added to appropriate exporter module**

## 🧪 Testing Practices

### 11. **Silent Failure Detection**
- ✅ **ALWAYS check for "0 computations" in test logs - this means NO execution happened**
- ✅ **Verify inference mode shows actual computations, not just "completed" status**
- ✅ **Training accuracy ~90% but inference ~8% indicates checkpoint loading issues**

### 12. **Performance Validation**
- ✅ **Benchmark against known baselines (e.g., IsaacGymEnvs)**
- ✅ **Measure both total throughput and per-environment performance**
- ✅ **Account for startup time in performance measurements**
- ✅ **Use consistent environment counts for fair comparisons**

## 📋 Pre-Commit Checklist

Before submitting any code changes, verify:

### Parameter Handling
- [ ] All exporters use `cls.get_node_parameters_batch()` 
- [ ] No direct access to `widgets_values` or `inputs`
- [ ] No fallback values masking missing data
- [ ] Clear error messages for missing parameters

### Base Classes
- [ ] All abstract methods throw `NotImplementedError`
- [ ] No "guessed" default implementations
- [ ] Clear documentation of what subclasses must implement

### File Organization  
- [ ] No files created in project root
- [ ] Exports go to correct `export_system/exports/` subdirectory
- [ ] Test files in appropriate `claude_scripts/` or `tests-dnne/` directories

### Templates
- [ ] Use async queue-based patterns
- [ ] Include proper error handling
- [ ] Use descriptive template variable names
- [ ] Provide complete template_vars defaults

### Testing
- [ ] Tests only marked complete when fully successful
- [ ] Check for silent failures (0 computations)
- [ ] Validate actual functionality, not just "no errors"

### Isaac Gym Integration
- [ ] Isaac Gym imported before torch
- [ ] Proper node import ordering in generated code
- [ ] Environment architecture uses clean class hierarchy

## 🔍 Common Anti-Patterns to Avoid

### 1. **Silent Fallbacks**
```python
# BAD - hides real problems
value = config.get("setting", "default")

# GOOD - reveals configuration issues  
if "setting" not in config:
    raise ValueError(f"Required setting 'setting' missing from config. Available: {list(config.keys())}")
value = config["setting"]
```

### 2. **Generic Error Messages**
```python
# BAD - provides no debugging information
raise ValueError("Invalid parameter")

# GOOD - specific diagnostic information
raise ValueError(f"Parameter 'num_envs' must be positive integer, got {num_envs} (type: {type(num_envs)})")
```

### 3. **Bypassing Helper Functions**
```python
# BAD - inconsistent data access
widget_values = node_data.get("widgets_values", [])
learning_rate = widget_values[0] if widget_values else 0.001

# GOOD - use universal parameter reader
param_specs = [{'name': 'learning_rate', 'default': 0.001}]
params = cls.get_node_parameters_batch(node_data, param_specs)
learning_rate = params['learning_rate']
```

### 4. **Optimistic Testing**
```python
# BAD - marking success without verification
def test_training():
    export_workflow()  # might fail
    # ... some code runs ...
    print("✅ Test passed!")  # but did it actually work?

# GOOD - verify actual success
def test_training():
    try:
        result = export_workflow()
        assert result.accuracy > 0.8, f"Low accuracy: {result.accuracy}"
        assert result.loss < 0.5, f"High loss: {result.loss}"
        print(f"✅ Test passed: accuracy={result.accuracy}, loss={result.loss}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
```

## 📚 Reference Documentation

- **Base Class Design**: See "Important Development Notes" in `/CLAUDE.md`
- **Export System**: See `export_system/templates/CLAUDE.md`
- **Universal Parameter Reader**: See `export_system/graph_exporter.py`
- **Queue Framework**: See `export_system/templates/base/queue_framework.py`
- **Testing Standards**: See "CRITICAL TESTING RULE" in `/CLAUDE.md`

## 🔄 Review Schedule

This checklist should be reviewed:
- **Before major feature development**
- **After encountering any silent failure or hard-to-debug issue**  
- **During code review process**
- **When onboarding new developers**
- **Monthly as part of code quality maintenance**

---

**Remember**: These rules exist because we've learned from painful debugging experiences. Following them prevents wasted time and ensures robust, maintainable code.