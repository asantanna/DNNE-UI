# Gotchas That Will Burn You 🔥

*These are the painful lessons that EVERYONE learns the hard way. Read this first and save yourself hours of frustration.*

## The Export Overwrite Trap 💀

**The Scenario**: You've spent an hour perfecting a fix in `exports/MyWorkflow/runner.py`. It works beautifully. You're about to commit... then someone (maybe even you) re-exports the workflow. YOUR CODE IS GONE. No warning. No backup. Just gone.

**Why It Happens**: Exports are generated from templates. They're meant to be disposable, like compiled binaries.

**The Fix**: ALWAYS fix the template, never the export:
```bash
# WRONG - This will be lost!
vi export_system/exports/MyWorkflow/runner.py

# RIGHT - This persists!
vi export_system/templates/framework/runner.tpl
```

**How to Remember**: Think of exports like .exe files - you don't hex-edit the binary, you fix the source code.

**Recovery Tip**: If you just lost work, check if your editor has a backup (vim's .swp files, VSCode's local history).

## Widget Index Confusion & Encoding Trap 🎲💀

**The Double Trap**: 
1. Widget indices skip (labels don't save values)
2. Widget encoding differs between UI export and programmatic export

**CRITICAL RULE**: NEVER access `widget_values` directly. The encoding is different between UI and programmatic export, causing silent failures that are nearly impossible to debug.

**Example of What Goes Wrong**:
```python
# UI shows: [Label] [Learning Rate: 0.01] [Batch Size: 32]
# But widget_values = [0.01, 32]  # No label!

# WRONG - Never do this!
learning_rate = widget_values[1]  # This is actually batch_size!

# ALSO WRONG - Still direct access!
learning_rate = widget_values[0]  # Might work in UI, fail in programmatic export!
```

**The ONLY Correct Way**: Use helper functions that handle encoding:
```python
# For single parameter
learning_rate = cls.get_node_parameter(node_data, 'learning_rate', default=0.01)

# For multiple parameters (preferred)
params = cls.get_node_parameters_batch(node_data, [
    {'name': 'learning_rate', 'default': 0.01},
    {'name': 'batch_size', 'default': 32}
])
```

**Why This Is Critical**: The same workflow will behave differently when exported from UI vs programmatically if you access `widget_values` directly. This creates "works on my machine" bugs that are nightmare fuel.

## ComfyUI Slot Corruption 🐛

**The Symptom**: Your export crashes with weird connection errors. Connections that should exist are missing or point to the wrong slots.

**Why It Happens**: ComfyUI sometimes corrupts the slot numbers in the workflow JSON during processing.

**The Workaround**: In `graph_exporter.py`, there's a `_fix_corrupted_slots()` method that uses the JSON representation as ground truth:
```python
# The real connections are in the JSON 'links' array
# ComfyUI's processed version in 'inputs' can be corrupted
link_data = original_workflow['links'][link_id]
```

**When You Hit This**: If connections are mysteriously wrong, check if the slot fixing code is running and using the original JSON data.

## Isaac Gym Import Order ☠️

**The Crash**: Segmentation fault, no error message, just death.

**The Cause**: PyTorch imported before Isaac Gym.

**The Rule**: Isaac Gym MUST be imported before PyTorch. Always. No exceptions.

```python
# CRASH - PyTorch first
import torch
import isaacgym  # Segfault!

# WORKS - Isaac Gym first  
import isaacgym
import torch
```

**Where This Matters**: 
- Runner.py templates
- Test scripts
- Any file using both libraries

## The Circular Bootstrap (Franka Trap) 🔄

**The Deadlock**: IsaacGymSim waits for action before sending observation. Network waits for observation before sending action. Nothing happens.

**Why It Happens**: Environments need an initial action to produce the first observation, but the controller needs an observation to produce an action.

**The Solution**: Bootstrap with `null_action`:
```python
class IsaacGymSimNode:
    async def run(self):
        # Bootstrap with null action FIRST
        null_action = [0.0] * self.action_size
        obs = self.env.step(null_action)
        await self.send_output(obs, "observation")
        
        # Now start normal loop
        while True:
            action = await self.get_input("action")
            obs = self.env.step(action)
            await self.send_output(obs, "observation")
```

## Async Task Churn Deadlock 🌀

**The Pattern That Kills**:
```python
# THIS CAUSES DEADLOCKS!
while True:
    tasks = [asyncio.create_task(q.get()) for q in queues]
    done, pending = await asyncio.wait(tasks, return_when=FIRST_COMPLETED)
    for task in pending:
        task.cancel()  # <-- Race condition here!
```

**Why It Deadlocks**: Constant task creation/cancellation creates timing windows where data arrives between cancellation and next creation.

**The Fix**: Use persistent listeners (MultiWaiter pattern):
```python
# Create listeners ONCE
listeners = {}
for name, queue in queues.items():
    listeners[name] = asyncio.create_task(listen_forever(queue))

# Never cancel them in normal operation
```

**The Lesson**: In async code, stable long-running tasks are better than creating/destroying tasks in loops.

## The asyncio.sleep(0) Mystery 🕐

**The Symptom**: Queues seem stuck, but adding `await asyncio.sleep(0)` magically fixes it.

**Why It Works**: `asyncio.sleep(0)` yields control to other tasks. Without it, a tight loop can monopolize the event loop.

**When You Need It**:
```python
# After sending many messages rapidly
for i in range(1000):
    await queue.put(data)
await asyncio.sleep(0)  # Let receivers process!

# Between queue operations that might deadlock
await output_queue.put(result)
await asyncio.sleep(0)  # Yield before waiting for input
input = await input_queue.get()
```

**But Don't Overuse**: Only add it when you see queue congestion or deadline issues. It's a code smell that might indicate a deeper issue.

## Dimension Mismatch Silent Failure 📏

**The Trap**: Your concat node works fine with 2D tensors, then silently produces garbage with 1D tensors.

**Why It's Silent**: PyTorch happily concatenates [32] and [32] to make [64], but that's concatenating batches, not features!

**The Rule**: Dimension 0 is ALWAYS batch, Dimension 1 is ALWAYS features. No exceptions.

**The Fix**: Fail fast and loud:
```python
if tensor.dim() < 2:
    raise ValueError(f"Expected 2D+ tensor [batch, features, ...], got {tensor.shape}")
```

## The Missing Context During Export 🕳️

**The Problem**: In the UI, nodes can access other nodes via context. During export, they can't.

**The Symptom**: Export crashes with "No context available" or "Can't find connected node".

**Why**: Export nodes don't execute - they generate code. They can't reach across the graph during template generation.

**The Solution**: Pass all needed info through parameters or use the registry:
```python
# WRONG - Tries to access other nodes during export
def prepare_template_vars(cls, node_id, node_data, connections):
    other_node = context.get_node(connected_id)  # FAILS!
    
# RIGHT - Use provided connections info
def prepare_template_vars(cls, node_id, node_data, connections, node_registry):
    connected_info = connections.get(node_id, {})
```

## The Telemetry Permission Trap 🔒

**The Surprise**: Your workflow runs fine locally, then "Permission Denied" in cloud/docker.

**The Cause**: Telemetry tries to write to paths that don't exist or aren't writable.

**The Fix**: Always check and create telemetry directories:
```python
telemetry_dir = Path("telemetry_data")
telemetry_dir.mkdir(parents=True, exist_ok=True)
```

**Better**: Use temp directories in constrained environments:
```python
import tempfile
telemetry_dir = Path(tempfile.gettempdir()) / "dnne_telemetry"
```

## The Widget Encoding Trap (THE #1 GOTCHA) ⚡

**The Most Insidious Bug**: Direct `widget_values` access works differently in UI export vs programmatic export.

**How It Manifests**:
- Your workflow works perfectly when exported from the UI
- The SAME workflow fails mysteriously when exported programmatically
- Or vice versa - works programmatically, fails from UI
- Values are wrong, misaligned, or missing

**The Root Cause**: The UI and programmatic export encode widget values differently. Direct array access `widget_values[0]` gets different data depending on export method.

**The Iron Rule**: NEVER, EVER access `widget_values` directly. Always use:
- `cls.get_node_parameter(node_data, 'param_name', default=value)`
- `cls.get_node_parameters_batch(node_data, param_specs)`

**This Is Non-Negotiable**: If you access `widget_values` directly, your code WILL break in production. Not might. WILL.

## Quick Reference Card

| If you see... | The gotcha is... | Do this... |
|--------------|------------------|------------|
| Lost code after export | Edited generated file | Fix template, re-export |
| Wrong widget values | Direct access + encoding | Use helper functions ONLY |
| Broken connections | Slot corruption | Check JSON workaround |
| Segfault with Isaac Gym | Import order | Import Isaac before PyTorch |
| Environment deadlock | Circular bootstrap | Use null_action pattern |
| Async deadlock | Task churn | Use persistent listeners |
| Queues stuck | Event loop monopoly | Add asyncio.sleep(0) |
| Concat wrong result | 1D tensors | Enforce 2D minimum |
| "No context" in export | Context doesn't exist | Use provided parameters |
| Permission denied | Telemetry paths | Create dirs or use temp |

## The Meta-Gotcha

**The Ultimate Trap**: Thinking you'll remember these gotchas without documentation.

**The Reality**: You'll hit the same gotcha three months from now and waste another hour.

**The Fix**: When you hit a new gotcha, add it here immediately. Future you will thank present you.