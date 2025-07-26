# Frontend Development Guide for DNNE-UI

This guide documents the process of making changes to the DNNE-UI frontend, including how to edit, build, and debug changes.

## Directory Structure

The frontend exists in two locations due to Windows/WSL constraints:

1. **Source Directory (Windows)**: `/mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend/`
   - This is the Git repository location
   - Edit files here for version control
   - Cannot run npm commands due to Windows filesystem permissions

2. **Build Directory (Linux)**: `/home/asantanna/DNNE-UI-Frontend/`
   - Clone of the repository in Linux filesystem
   - Run npm commands and builds here
   - Must sync changes from source directory

## Development Workflow

### 1. Making Code Changes

Always edit files in BOTH directories to keep them synchronized:

```bash
# Edit in Windows directory first (for Git tracking)
/mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend/src/...

# Then copy to Linux directory for building
cp /mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend/src/path/to/file.ts /home/asantanna/DNNE-UI-Frontend/src/path/to/file.ts
```

### 2. Building the Frontend

The frontend MUST be built in the Linux directory:

```bash
# Navigate to Linux build directory
cd /home/asantanna/DNNE-UI-Frontend

# Build the project
npm run build
```

**Note**: The user typically runs the build command manually. Wait for them to confirm "build complete" before proceeding.

#### Automated Build Script

A helper script is available to automate the build and sync process:

```bash
# From the DNNE-UI directory
bash /mnt/e/ALS-Projects/DNNE/DNNE-UI/build_frontend.sh
```

This script:
1. Builds the frontend in the Linux directory
2. Automatically syncs the dist folder to the Windows directory
3. Provides build status feedback

### 3. Deploying Changes

After building, copy the compiled files back to the Windows directory:

```bash
# The user will typically use rsync
rsync -av /home/asantanna/DNNE-UI-Frontend/dist/ /mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend/dist/
```

**Note**: If using the `build_frontend.sh` script, this step is done automatically.

### 4. Testing Changes

1. Restart the DNNE server (user must do this on Windows side)
2. Hard refresh the browser (Ctrl+Shift+R) or use incognito mode to avoid cache issues
3. Check browser console for any errors or debug messages

## Common Issues and Solutions

### TypeScript Compilation Errors

Common errors and fixes:

1. **"X is declared but never used"**
   - Prefix unused variables with underscore: `_variableName`
   - Or remove the import/variable if truly unused

2. **"Cannot find module"**
   - Ensure all imports use correct paths
   - Check that dependencies are installed in package.json

### Browser Cache Issues

The browser aggressively caches JavaScript files. Solutions:

1. Use incognito/private browsing mode
2. Hard refresh multiple times (Ctrl+Shift+R)
3. Clear browser cache completely
4. Add cache-busting query parameters to script URLs

### Directory Confusion

Always verify which directory you're working in:

```bash
# Check current directory
pwd

# Ensure frontend changes are in the Linux directory before building
ls -la /home/asantanna/DNNE-UI-Frontend/src/composables/widgets/
```

## Debugging Frontend Changes

### Browser Developer Tools

1. Open browser console (F12)
2. Set breakpoints in Sources tab
3. Use console.log() for debugging (remove before final commit)
4. Check Network tab to ensure new files are loaded

### Vue DevTools

Install Vue DevTools browser extension for debugging Vue components:
- Inspect component state
- Track reactive data changes
- Monitor events

### Common Widget Issues

When working with ComfyUI widgets:

1. **Widget not updating**: Check that widget.name matches the key in configuration
2. **Callback not firing**: Verify the widget type and initialization
3. **Node type detection**: Node type can be in `node.type`, `node.comfyClass`, or `node.constructor.name`

## File Synchronization Commands

Keep these commands handy:

```bash
# Copy single file from Windows to Linux
cp /mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend/src/path/to/file.ts /home/asantanna/DNNE-UI-Frontend/src/path/to/file.ts

# Copy directory recursively
cp -r /mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend/src/composables/ /home/asantanna/DNNE-UI-Frontend/src/composables/

# Compare files between directories
diff /home/asantanna/DNNE-UI-Frontend/src/file.ts /mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend/src/file.ts

# Sync built files back to Windows (user typically does this)
rsync -av /home/asantanna/DNNE-UI-Frontend/dist/ /mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend/dist/
```

## Key Widget Development Concepts

### DNNE Combo Widget

The custom DNNE combo widget adds onChange callbacks for updating multiple nodes:

1. Located in: `src/composables/widgets/useDNNEComboWidget.ts`
2. Registered in: `src/scripts/widgets.ts`
3. Used for: Task selection in IsaacGymEnvs node

### Widget Value Updates

For widgets to update properly:

1. Widget name must match the configuration key
2. Use `widget.value = newValue` to update
3. Call `widget.callback(newValue)` if needed
4. The graph must be marked dirty: `app.graph.setDirtyCanvas(true, true)`
5. **CRITICAL**: Call `app.graph.change()` to notify LiteGraph of changes for proper saving
6. Update ChangeTracker state: `activeWorkflow.changeTracker.checkState()`

### Control After Generate Widgets

Special ComfyUI widgets that appear after certain inputs:

1. Named `control_after_generate` (not the display text)
2. Often linked to other widgets via `linkedWidgets` property
3. Common for seed controls and similar randomization options

## Helper Scripts

Several helper scripts are available to streamline the development workflow:

### build_frontend.sh

Location: `/mnt/e/ALS-Projects/DNNE/DNNE-UI/build_frontend.sh`

This script automates the entire build and sync process:
- Builds the frontend in the Linux directory
- Syncs the dist folder to the Windows directory  
- Provides clear status messages

Usage:
```bash
bash /mnt/e/ALS-Projects/DNNE/DNNE-UI/build_frontend.sh
```

### rsync_frontend.sh

Location: `/mnt/e/ALS-Projects/DNNE/DNNE-UI/rsync_frontend.sh`

**Note**: This script currently syncs to `/mnt/e/ALS-Projects/DNNE/DNNE-UI/web/` which is incorrect. It should sync to `/mnt/e/ALS-Projects/DNNE/DNNE-UI-Frontend/dist/`. Use `build_frontend.sh` instead or manually rsync to the correct location.

## Environment Setup

Before building, ensure the conda environment is activated:

```bash
source /home/asantanna/miniconda/bin/activate DNNE_PY38
```

## Build Commands Reference

```bash
# Development build with hot reload (if needed)
npm run dev

# Production build
npm run build

# Type checking
npm run typecheck

# Linting
npm run lint

# Format code
npm run format
```

## Important Notes

1. **Always edit in both directories** to maintain synchronization
2. **Build only in Linux directory** due to filesystem permissions
3. **Wait for user confirmation** of build completion
4. **Clear browser cache** when testing changes
5. **Remove console.log statements** before finalizing code
6. **Test with multiple environments** when working on environment-specific features

## Recent Gotchas

1. **Seed Control Widget**: The widget is named `control_after_generate`, not `seed_control`
2. **Node Type Detection**: Must check multiple properties (`node.type`, `node.comfyClass`, etc.)
3. **Widget Updates**: The check `!(widget.name in widgetValues)` skips widgets whose names don't match config keys
4. **OmegaConf Interpolations**: Backend must resolve `${...}` strings to concrete values
5. **Widget Persistence**: Without calling `app.graph.change()`, programmatically updated widget values won't save
6. **Modified Indicator**: The "modified" tab indicator may disappear when clicking canvas without proper ChangeTracker update

## Example: Fixing Widget Persistence

When programmatically updating widget values (e.g., from server config), ensure persistence:

```typescript
// Update widget value
widget.value = newValue

// Call callback if exists
if (widget.callback) {
  widget.callback(newValue)
}

// Mark graph as dirty
app.graph.setDirtyCanvas(true, true)

// CRITICAL: Notify LiteGraph of changes
app.graph.change()

// Update ChangeTracker immediately
const workflowStore = useWorkflowStore()
const activeWorkflow = workflowStore.activeWorkflow
if (activeWorkflow?.changeTracker) {
  activeWorkflow.changeTracker.checkState()
}
```

This guide should help avoid the common pitfalls encountered during frontend development.