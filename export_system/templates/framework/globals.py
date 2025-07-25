"""
Global state management for DNNE exported workflows.
Provides centralized configuration and adaptive yielding.
"""

import asyncio
import time
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class YieldStats:
    """Statistics for adaptive yielding performance"""
    total_yields: int = 0
    total_yield_time: float = 0.0
    yield_overhead_ns: float = 0.0
    last_yield_time: float = 0.0
    
    def update(self, yield_duration: float):
        self.total_yields += 1
        self.total_yield_time += yield_duration
        self.yield_overhead_ns = (self.total_yield_time / self.total_yields) * 1e9 if self.total_yields > 0 else 0
        self.last_yield_time = time.time()


@dataclass 
class NodeMetrics:
    """Metrics for individual node execution"""
    last_execution: float = 0.0
    execution_count: int = 0
    total_wait_time: float = 0.0
    
    @property
    def starvation_time(self) -> float:
        """Time since last execution"""
        return time.time() - self.last_execution if self.last_execution > 0 else 0.0


class Global:
    """
    Central configuration and state management for DNNE workflows.
    
    Usage:
        from framework.globals import Global as g
        
        if g.inference_mode:
            # Skip training
            
        await g.AdaptiveYield()
    """
    
    # === Execution Modes ===
    inference_mode: bool = False
    training_mode: bool = True
    visual_mode: bool = False
    
    # === Debug Settings ===
    verbose: bool = False
    profiling: bool = False
    debug: bool = False
    fixed_seed: Optional[int] = None
    
    # === Performance Settings ===
    device: str = "cuda"  # Will be set properly at runtime
    no_yield: bool = False  # Disable all yielding for performance comparison
    
    # === Paths ===
    save_checkpoint_dir: Optional[Path] = None
    load_checkpoint_dir: Optional[Path] = None
    export_dir: Path = Path(".")
    
    # === Adaptive Yielding Settings ===
    # Thresholds in seconds
    CRITICAL_STARVATION_THRESHOLD: float = 0.1  # 100ms
    WARNING_STARVATION_THRESHOLD: float = 0.05  # 50ms
    
    # Yield frequencies
    DEFAULT_YIELD_FREQUENCY: int = 100  # Yield every N iterations
    
    # === Private State ===
    _yield_disabled: int = 0  # Counter for nested no_yield contexts
    _yield_stats: YieldStats = YieldStats()
    _node_metrics: Dict[str, NodeMetrics] = {}
    _total_queued: int = 0
    _initialized: bool = False
    _logger: Optional[logging.Logger] = None
    
    @classmethod
    def initialize(cls, **kwargs):
        """
        Initialize global settings from command-line arguments or config.
        
        Args:
            **kwargs: Settings to override defaults
        """
        # Set execution modes
        cls.inference_mode = kwargs.get('inference_mode', False)
        cls.training_mode = not cls.inference_mode
        cls.visual_mode = kwargs.get('visual_mode', False)
        
        # Debug settings
        cls.verbose = kwargs.get('verbose', False)
        cls.profiling = kwargs.get('profiling', False)
        cls.debug = kwargs.get('debug', False)
        cls.fixed_seed = kwargs.get('fixed_seed', None)
        
        # Performance settings
        cls.no_yield = kwargs.get('no_yield', False)
        
        # Paths
        if kwargs.get('save_checkpoint_dir'):
            cls.save_checkpoint_dir = Path(kwargs['save_checkpoint_dir'])
        if kwargs.get('load_checkpoint_dir'):
            cls.load_checkpoint_dir = Path(kwargs['load_checkpoint_dir'])
        
        # Device - lazy import torch
        device = kwargs.get('device', 'auto')
        if device == 'auto':
            try:
                import torch
                cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError as e:
                raise RuntimeError(
                    f"PyTorch is required but not installed: {e}\n"
                    f"Please install PyTorch before running DNNE workflows.\n"
                    f"Visit https://pytorch.org/get-started/locally/ for installation instructions."
                )
        else:
            cls.device = device
        
        # Logger
        cls._logger = logging.getLogger('Global')
        
        cls._initialized = True
        
        # Log initialization
        if cls.verbose:
            cls._logger.info(f"Global initialized: mode={'inference' if cls.inference_mode else 'training'}, "
                           f"device={cls.device}, verbose={cls.verbose}")
        
        if cls.no_yield:
            cls._logger.info("Adaptive yielding DISABLED - running at full speed for performance comparison")
    
    @classmethod
    async def async_adaptive_yield(cls):
        """
        Async adaptive yield - use in async functions.
        Automatically adjusts yield duration based on system metrics.
        """
        # Fast path - no yield if disabled by context or command line
        if cls._yield_disabled > 0 or cls.no_yield:
            return
            
        start_time = time.perf_counter()
        
        # Compute adaptive delay
        delay = cls._compute_adaptive_delay()
        
        # Perform yield
        await asyncio.sleep(delay)
        
        # Update statistics
        end_time = time.perf_counter()
        yield_duration = end_time - start_time
        cls._yield_stats.update(yield_duration)
        
        # Log if excessive yield time
        if cls.debug and yield_duration > 0.01:  # 10ms
            cls._logger.warning(f"Long yield detected: {yield_duration*1000:.2f}ms")
    
    @classmethod
    def sync_adaptive_yield(cls):
        """
        Sync adaptive yield - use in synchronous functions.
        Uses event loop internals to yield from sync code.
        
        Warning: This uses private asyncio APIs (_run_once) and may break
        in future Python versions. Use only when async is not possible.
        """
        # Fast path - no yield if disabled by context or command line
        if cls._yield_disabled > 0 or cls.no_yield:
            return
        
        start_time = time.perf_counter()
        
        # Get event loop - fail if none
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError(
                "sync_adaptive_yield() called but no event loop is running! "
                "All DNNE workflows must run within an async context. "
                "If you're in an async function, use async_adaptive_yield() instead."
            )
        
        # Compute adaptive delay
        delay = cls._compute_adaptive_delay()
        
        if delay == 0:
            # Quick yield - just run one iteration of event loop
            loop._run_once()
        else:
            # Timed delay using event loop timer
            done = False
            
            def set_done():
                nonlocal done
                done = True
            
            # Schedule callback after delay
            loop.call_later(delay, set_done)
            
            # Run event loop until timer fires
            while not done:
                loop._run_once()
        
        # Update statistics
        end_time = time.perf_counter()
        yield_duration = end_time - start_time
        cls._yield_stats.update(yield_duration)
        
        # Log if excessive yield time
        if cls.debug and yield_duration > 0.01:  # 10ms
            cls._logger.warning(f"Long sync yield detected: {yield_duration*1000:.2f}ms")
    
    @classmethod
    def _compute_adaptive_delay(cls) -> float:
        """
        Compute appropriate yield delay based on system metrics.
        
        Returns:
            Delay in seconds (0 for minimal yield, up to 0.01 for aggressive yielding)
        """
        # If not tracking metrics, use minimal yield
        if not cls._node_metrics:
            return 0.0
            
        # Find maximum starvation time
        max_starvation = max(
            (metrics.starvation_time for metrics in cls._node_metrics.values()),
            default=0.0
        )
        
        # Adaptive delay based on starvation
        if max_starvation > cls.CRITICAL_STARVATION_THRESHOLD:
            return 0.01  # 10ms - aggressive yielding
        elif max_starvation > cls.WARNING_STARVATION_THRESHOLD:
            return 0.001  # 1ms - moderate yielding  
        elif cls._total_queued > 10:  # High queue pressure
            return 0.0001  # 0.1ms - light yielding
        else:
            return 0.0  # Minimal yielding
    
    @classmethod
    @contextmanager
    def no_yield(cls):
        """
        Context manager to temporarily disable adaptive yielding.
        Supports nested contexts.
        
        Usage:
            with Global.no_yield():
                # Critical section without yields
                fast_computation()
        """
        cls._yield_disabled += 1
        try:
            yield
        finally:
            cls._yield_disabled -= 1
    
    @classmethod
    def update_node_execution(cls, node_id: str):
        """
        Update execution metrics for a node.
        Called by nodes when they execute.
        
        Args:
            node_id: Unique identifier of the executing node
        """
        if node_id not in cls._node_metrics:
            cls._node_metrics[node_id] = NodeMetrics()
            
        metrics = cls._node_metrics[node_id]
        current_time = time.time()
        
        # Update wait time
        if metrics.last_execution > 0:
            wait_time = current_time - metrics.last_execution
            metrics.total_wait_time += wait_time
            
        metrics.last_execution = current_time
        metrics.execution_count += 1
    
    @classmethod
    def update_queue_pressure(cls, total_queued: int):
        """
        Update total items queued across all nodes.
        
        Args:
            total_queued: Total number of items in all queues
        """
        cls._total_queued = total_queued
    
    @classmethod
    def get_yield_stats(cls) -> Dict[str, Any]:
        """
        Get statistics about adaptive yielding performance.
        
        Returns:
            Dictionary with yield statistics
        """
        stats = cls._yield_stats
        return {
            'total_yields': stats.total_yields,
            'total_yield_time_s': stats.total_yield_time,
            'avg_yield_time_us': (stats.total_yield_time / stats.total_yields * 1e6) if stats.total_yields > 0 else 0,
            'yield_overhead_ns': stats.yield_overhead_ns,
            'nodes_tracked': len(cls._node_metrics),
            'max_starvation_ms': max(
                (m.starvation_time * 1000 for m in cls._node_metrics.values()),
                default=0.0
            )
        }
    
    @classmethod
    def reset_metrics(cls):
        """Reset all performance metrics (useful for benchmarking)"""
        cls._yield_stats = YieldStats()
        cls._node_metrics.clear()
        cls._total_queued = 0
    
    @classmethod
    def system_healthy(cls) -> bool:
        """
        Quick check if system is operating normally.
        
        Returns:
            True if no significant starvation detected
        """
        if not cls._node_metrics:
            return True
            
        max_starvation = max(
            (metrics.starvation_time for metrics in cls._node_metrics.values()),
            default=0.0
        )
        
        return max_starvation < cls.WARNING_STARVATION_THRESHOLD


# Backward compatibility helpers
def _sync_with_builtins():
    """
    Maintain backward compatibility by syncing with builtins.
    This will be removed in future versions.
    """
    import builtins
    
    # Sync key attributes
    builtins.INFERENCE_MODE = Global.inference_mode
    builtins.VERBOSE = Global.verbose
    builtins.FIXED_SEED = Global.fixed_seed
    builtins.SAVE_CHECKPOINT_DIR = str(Global.save_checkpoint_dir) if Global.save_checkpoint_dir else None
    builtins.LOAD_CHECKPOINT_DIR = str(Global.load_checkpoint_dir) if Global.load_checkpoint_dir else None
    builtins.DNNE_PROFILING = Global.profiling


# Auto-sync on import (temporary for migration)
_sync_with_builtins()