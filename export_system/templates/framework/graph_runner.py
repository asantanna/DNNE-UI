# Graph Runner - Manages and executes the node graph

import asyncio
import logging
from typing import Dict, Any, List, Optional

from .base_nodes import QueueNode
from .exceptions import CauseExitException
from .globals import Global as g, dnne_logging

# Queue subsystem logger
queue_logger = dnne_logging.getLogger("queue")


class GraphRunner:
    """Manages and runs the complete node graph"""
    
    def __init__(self):
        self.nodes: Dict[str, QueueNode] = {}
        self.tasks: List[asyncio.Task] = []
        # GraphRunner uses general logger for high-level operations
        self.logger = logging.getLogger(__name__)
        
        # Exit tracking for smart checkpoint saves
        self.exit_reason = None
        self.exit_code = 0  # Default exit code
        self.has_completion_conditions = False
    
    def add_node(self, node: QueueNode):
        """Add a node to the graph"""
        self.nodes[node.node_id] = node
        self.logger.debug(f"Added node: {node.node_id}")
    
    def wire_nodes(self, connections: List[tuple]):
        """Wire nodes together: (from_id, output, to_id, input)"""
        # Build a dictionary to track connections for each node
        node_connections = {}
        
        for from_id, output_name, to_id, input_name in connections:
            from_node = self.nodes[from_id]
            to_node = self.nodes[to_id]
            
            # Subscribe to_node's input queue to from_node's output
            from_node.output_subscribers[output_name].append(
                to_node.input_queues[input_name]
            )
            self.logger.debug(f"Connected {from_id}.{output_name} -> {to_id}.{input_name}")
            
            # Track connections for the receiving node
            if to_id not in node_connections:
                node_connections[to_id] = {}
            if input_name not in node_connections[to_id]:
                node_connections[to_id][input_name] = []
            node_connections[to_id][input_name].append({
                "from_node": from_id,
                "from_output": output_name
            })
        
        # Inform nodes about their connections
        for node_id, connections_dict in node_connections.items():
            node = self.nodes[node_id]
            if hasattr(node, 'set_connections'):
                self.logger.debug(f"Setting connections for node {node_id}: {connections_dict}")
                node.set_connections(connections_dict)
    
    def _detect_completion_conditions(self):
        """Detect if the workflow has any defined completion conditions"""
        # Check for EpochTracker (indicates supervised learning with defined epochs)
        has_epoch_tracker = any("EpochTracker" in node.__class__.__name__ for node in self.nodes.values())
        
        # Check for PPOTrainerNode with max_epochs (reinforcement learning with defined epochs)
        has_ppo_trainer_with_epochs = any(
            "PPOTrainerNode" in node.__class__.__name__ and hasattr(node, 'max_epochs')
            for node in self.nodes.values()
        )
        
        # Check for timeout specified at runtime (duration parameter)
        # This will be checked in the run method
        
        self.has_completion_conditions = has_epoch_tracker or has_ppo_trainer_with_epochs
        return self.has_completion_conditions
    
    async def run(self, duration: Optional[float] = None):
        """Run all nodes"""
        self.logger.debug("Starting graph execution")
        
        # Detect completion conditions
        self._detect_completion_conditions()
        if duration is not None:
            self.has_completion_conditions = True  # Timeout is a completion condition
        
        # Check if we're in inference mode
        inference_mode = g.inference_mode
        
        try:
            if inference_mode:
                self.logger.debug("Running in INFERENCE mode - gradients disabled")
                # Import torch only if in inference mode
                import torch
                
                # Run in no_grad context for inference
                with torch.no_grad():
                    await self._run_graph(duration)
            else:
                await self._run_graph(duration)
        finally:
            # Handle checkpoint saves before final cleanup
            await self._handle_exit_checkpoints()
    
    async def _run_graph(self, duration: Optional[float] = None):
        """Internal method to run the graph"""
        # System ready event should already be initialized in runner.py
        if g._system_ready is None:
            raise RuntimeError("System ready event not initialized! runner.py must call Global.init_system_ready() before creating nodes!")
        
        # Simple check that all nodes registered
        expected_count = len(self.nodes)
        registered_count = len(g._registered_nodes)
        if registered_count < expected_count:
            raise RuntimeError(f"Not all nodes registered! Expected {expected_count}, got {registered_count}")
        elif registered_count > expected_count:
            raise RuntimeError(f"Too many nodes registered! Expected {expected_count}, got {registered_count}")
        
        # Start all nodes (they will report ready and wait at the barrier)
        for node in self.nodes.values():
            # Use the wrapper method instead of run() directly
            task = asyncio.create_task(node._call_run_when_ready())
            self.tasks.append(task)
        
        # Give nodes a moment to report ready
        await asyncio.sleep(0.01)
        
        # Signal that connections are ready - this releases all nodes
        g.report_connections_ready()
        self.logger.info("Starting graph execution")
        
        # Start heartbeat task in debug mode
        heartbeat_task = None
        if g.debug:
            heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
            self.logger.debug("💓 Started heartbeat monitor (5s interval)")
        
        # Setup timeout watchdog if duration specified
        timeout_future = None
        if duration:
            import threading
            import time
            
            # Create future for timeout communication
            timeout_future = asyncio.Future()
            
            # Capture the current event loop before starting the thread
            loop = asyncio.get_running_loop()
            
            def timeout_watchdog():
                """Watchdog thread that sets exception on timeout"""
                time.sleep(duration)
                if not timeout_future.done():
                    # Set exception on the future
                    exc = CauseExitException(
                        node_id="timeout_watchdog",
                        message=f"Timeout after {duration}s",
                        exit_code=0
                    )
                    # Use the captured loop reference
                    loop.call_soon_threadsafe(timeout_future.set_exception, exc)
            
            # Start the watchdog thread
            threading.Thread(
                target=timeout_watchdog,
                name="DNNE-Timeout-Watchdog",
                daemon=True
            ).start()
            self.logger.debug(f"⏱️  Started timeout watchdog: {duration}s")
        
        try:
            # Include timeout_future in gather if we have one
            if timeout_future:
                # This will raise CauseExitException if timeout occurs
                await asyncio.gather(
                    asyncio.gather(*self.tasks, return_exceptions=False),
                    timeout_future,
                    return_exceptions=False
                )
            else:
                # No timeout, just wait for tasks
                await asyncio.gather(*self.tasks, return_exceptions=False)
            
            self.exit_reason = "tasks_complete"
            
        except CauseExitException as e:
            # Handle exit requests (including timeout)
            if e.node_id == "timeout_watchdog":
                self.exit_reason = "timeout"
            else:
                self.exit_reason = "exit_requested"
            self.exit_code = e.exit_code
            self.logger.info(f"Exit (reason={self.exit_reason}): {e.message} (code={e.exit_code})")
        except KeyboardInterrupt:
            self.exit_reason = "keyboard_interrupt"
            self.logger.info("Interrupted by user")
        finally:
            # Cancel heartbeat task if running
            if heartbeat_task and not heartbeat_task.done():
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)
            
            # Cancel all tasks
            for task in self.tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for cancellation
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.logger.debug("All nodes stopped")
    
    
    async def _heartbeat_monitor(self):
        """Debug heartbeat that shows node activity every 5 seconds"""
        import time
        
        while True:
            try:
                await asyncio.sleep(5.0)
                
                # Get activity stats from Global
                stats = g.get_node_activity_stats()
                
                # Get queue pressure
                total_queued = 0
                queue_info = []
                for node_id, node in self.nodes.items():
                    for input_name, queue in node.input_queues.items():
                        size = queue.qsize()
                        total_queued += size
                        if size > 0:
                            queue_info.append(f"{node_id}.{input_name}:{size}")
                
                # Get compute rates for active nodes
                active_rates = []
                for node_id, _ in stats['active_nodes'][:3]:
                    if node_id in self.nodes:
                        node = self.nodes[node_id]
                        if node.compute_count > 0:
                            # Simple rate calculation (would need more sophisticated tracking for real rate)
                            active_rates.append(f"{node_id}({node.compute_count})")
                
                # Format idle nodes
                idle_info = []
                for node_id, idle_time in stats['idle_nodes'][:2]:
                    idle_info.append(f"{node_id}({idle_time:.1f}s)")
                
                # Build heartbeat message
                msg_parts = [
                    f"💓 Heartbeat: {stats['active_count']}/{stats['total_count']} nodes active"
                ]
                
                if total_queued > 0:
                    msg_parts.append(f"Queued: {total_queued} msgs")
                    if queue_info:
                        msg_parts.append(f"Queues: {', '.join(queue_info[:3])}")
                
                if active_rates:
                    msg_parts.append(f"Active: {', '.join(active_rates)}")
                
                if idle_info:
                    msg_parts.append(f"Idle: {', '.join(idle_info)}")
                
                self.logger.debug(" | ".join(msg_parts))
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
    
    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get execution statistics"""
        return {
            node_id: {
                "compute_count": node.compute_count,
                "last_compute_time": node.last_compute_time,
                "running": node.running
            }
            for node_id, node in self.nodes.items()
        }
    
    async def _handle_exit_checkpoints(self):
        """Handle checkpoint saves on exit based on exit reason and completion conditions"""
        
        # Skip if in inference mode
        inference_mode = g.inference_mode
        if inference_mode:
            self.logger.info("💾 Skipping exit checkpoints - inference mode")
            return
        
        # Determine if we should save checkpoint on exit
        should_save = False
        reason_message = ""
        
        if self.exit_reason == "timeout":
            should_save = True
            reason_message = "Training timeout reached - saving final checkpoint"
            
        elif self.exit_reason == "exit_requested":
            should_save = True  
            reason_message = "Exit requested - saving final checkpoint"
            
        elif self.exit_reason == "keyboard_interrupt":
            if self.has_completion_conditions:
                should_save = False
                reason_message = "Keyboard interrupt with defined completion conditions - not saving checkpoint"
            else:
                should_save = True
                reason_message = "Keyboard interrupt on indefinite run - saving checkpoint"
                
        elif self.exit_reason == "indefinite_run":
            should_save = True
            reason_message = "Indefinite run stopped - saving checkpoint"
            
        # Log the decision
        if should_save:
            self.logger.info(f"💾 {reason_message}")
        else:
            self.logger.info(f"🚫 {reason_message}")
        
        # Trigger checkpoint saves on eligible nodes
        if should_save:
            saved_count = 0
            for node in self.nodes.values():
                if hasattr(node, 'save_checkpoint_on_exit'):
                    try:
                        success = await node.save_checkpoint_on_exit(self.exit_reason)
                        if success:
                            saved_count += 1
                    except Exception as e:
                        self.logger.error(f"Failed to save exit checkpoint for {node.node_id}: {e}")
                        
            if saved_count > 0:
                self.logger.info(f"💾 Saved exit checkpoints for {saved_count} nodes")
            else:
                self.logger.info("💾 No nodes saved exit checkpoints")