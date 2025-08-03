# DNNE Agent Integration Tasks

This document tracks the implementation of DNNE Agent integration for remote workflow deployment.

**Reference**: See `dnne-agent-integration.md` for detailed architecture and message protocols.

## Current Status
- [x] Phase 1: Frontend UI Changes ✅ 2025-08-02
- [ ] Phase 2: Backend Integration  
- [ ] Phase 3: Testing & Polish

## Phase 1: Frontend UI Changes

### 1.1 Update Export Dropdown ✅
**File**: `/DNNE-UI-Frontend/src/components/actionbar/ComfyQueueButton.vue`
- [x] Replace queue mode items with export target dropdown
- [x] Add "📍 Local" as first item (default)
- [x] Add dynamic client list items "🖥️ {hostname}"
- [x] Remove queue mode logic (instant, on change, etc.)
- [x] Store selected target in component state

### 1.2 Create Agent Store ✅
**File**: `/DNNE-UI-Frontend/src/stores/agentStore.ts` (new)
- [x] Create Pinia store for agent state
- [x] Add client list: `{id, hostname, platform, connected_at}`
- [x] Add connection status: connected/disconnected/connecting
- [x] Add methods: updateClients, addClient, removeClient
- [x] Mock data for testing: ["wsl-machine", "ubuntu-box"]

### 1.3 Add Status Bar Component ✅
**File**: `/DNNE-UI-Frontend/src/components/actionbar/AgentStatusBar.vue` (new)
- [x] Create status bar component
- [x] Show connection indicator with colors (🟢🟡🔴⚫)
- [x] Display client count
- [x] Show active workflows count
- [x] Position below main toolbar

### 1.4 Update Main Actionbar ✅
**File**: `/DNNE-UI-Frontend/src/components/actionbar/ComfyActionbar.vue`
- [x] Import and include AgentStatusBar component
- [x] Adjust layout to accommodate status bar

### 1.5 Modify Export Request ✅
**File**: `/DNNE-UI-Frontend/src/scripts/api.ts`
- [x] Add `export_target` field to QueuePromptRequestBody interface
- [x] Include `export_target` from workspaceStore in request body
- [x] Add `run_after_export` field to interface (for future use)

## Phase 2: Backend Integration ✅

### 2.1 DNNE Server Agent Client ✅
**File**: `/DNNE-UI/server.py`
- [x] Add agent WebSocket client connection to port 8767
- [x] Handle connection/reconnection logic
- [x] Process incoming messages (server_state, client updates)
- [x] Cache client list for API endpoint

### 2.2 Agent Server Startup ✅
**File**: `/DNNE-UI/main.py`
- [x] Check if agent server running before DNNE startup
- [x] Start agent server subprocess if needed
- [x] Add retry logic with timeout

### 2.3 Client List API Endpoint ✅
**File**: `/DNNE-UI/server.py`
- [x] Add GET `/api/agent/clients` endpoint
- [x] Return cached client list from agent connection
- [x] Include connection status

### 2.4 Export Workflow Handler ✅
**File**: `/DNNE-UI/server.py` (modify `/prompt` endpoint)
- [x] Check `export_target` in request
- [x] If local: current behavior
- [x] If remote: package files and send via agent

### 2.5 WebSocket Message Forwarding ✅
**File**: `/DNNE-UI/server.py`
- [x] Forward agent updates to UI WebSocket
- [x] Handle workflow status messages
- [x] Forward telemetry data

## Phase 3: Testing & Polish

### 3.1 Frontend Testing
- [ ] Mock agent connection states
- [ ] Test dropdown updates on client connect/disconnect
- [ ] Verify status bar indicators
- [ ] Test export target selection persistence

### 3.2 Integration Testing
- [ ] Test agent server auto-start
- [ ] Test client list synchronization
- [ ] Test remote export workflow
- [ ] Test "run after export" functionality
- [ ] Test error handling (disconnections, failures)

### 3.3 UI Polish
- [ ] Add loading states during export
- [ ] Show progress for remote transfers
- [ ] Clear error messages
- [ ] Tooltips for connection states

## Key Decisions
- Export button renamed from "Queue Prompt" (already done)
- Using UI port (8767) exclusively for DNNE-agent communication
- Agent server persists when DNNE exits
- Local export uses `export_target: "local"`
- Status bar shows real-time connection state

## Testing Checklist
- [ ] Local export still works
- [ ] Remote export creates package correctly
- [ ] Client list updates dynamically
- [ ] Connection indicators accurate
- [ ] Error messages helpful
- [ ] No regression in existing features

## Known Issues/Blockers
- None yet

## Next Steps
1. Start with mock agent store (1.2)
2. Update ComfyQueueButton dropdown (1.1)
3. Get UI feedback before backend work