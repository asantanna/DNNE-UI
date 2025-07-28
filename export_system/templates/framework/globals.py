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

# Thread-safe yielding for sync code
from .globals_threadsafe import thread_safe_sync_adaptive_yield


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
class ConcurrencyStats:
    """Statistics for concurrent execution tracking"""
    ppo_time: float = 0.0  # Time spent in PPO subgraph
    non_ppo_time: float = 0.0  # Time spent in non-PPO subgraph
    last_yield_start: float = 0.0  # When the last yield started
    last_yield_time: float = 0.0  # When the last yield ended
    in_ppo_context: bool = False  # Whether we're currently in PPO execution
    
    def start_yield(self):
        """Called when sync_adaptive_yield is entered"""
        current_time = time.time()
        
        # Time between yields is PPO execution time
        if self.last_yield_time > 0:
            ppo_duration = current_time - self.last_yield_time
            if ppo_duration > 0:
                self.ppo_time += ppo_duration
        
        self.last_yield_start = current_time
        self.in_ppo_context = True
    
    def end_yield(self, yield_duration: float):
        """Called when sync_adaptive_yield completes"""
        # Time during yield is non-PPO (MNIST) execution time
        self.non_ppo_time += yield_duration
        self.last_yield_time = time.time()
        self.in_ppo_context = False
    
    def get_balance_percentage(self):
        """Get percentage time spent in PPO vs non-PPO"""
        total = self.ppo_time + self.non_ppo_time
        if total == 0:
            return (0.0, 0.0)
        return (self.ppo_time / total * 100, self.non_ppo_time / total * 100)


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
    headless_mode: bool = False
    
    # === Debug Settings ===
    verbose: bool = False
    profiling: bool = False
    debug: bool = False
    fixed_seed: Optional[int] = None
    
    # === Training Settings ===
    epochs_override: Optional[int] = None
    
    # === Performance Settings ===
    device: str = "cuda"  # Will be set properly at runtime
    _device_raw: str = "auto"  # Raw device setting before resolution
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
    _concurrency_stats: ConcurrencyStats = ConcurrencyStats()
    _node_metrics: Dict[str, NodeMetrics] = {}
    _total_queued: int = 0
    _initialized: bool = False
    _logger: Optional[logging.Logger] = None
    
    # === Node-specific Configuration ===
    node_configs: Dict[str, Dict[str, Any]] = {}  # node_id -> {config_key: value}
    
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
        cls.headless_mode = kwargs.get('headless_mode', False)
        
        # Debug settings
        cls.verbose = kwargs.get('verbose', False)
        cls.profiling = kwargs.get('profiling', False)
        cls.debug = kwargs.get('debug', False)
        cls.fixed_seed = kwargs.get('fixed_seed', None)
        
        # Training settings
        cls.epochs_override = kwargs.get('epochs_override', None)
        
        # Performance settings
        cls.no_yield = kwargs.get('no_yield', False)
        
        # Paths
        if kwargs.get('save_checkpoint_dir'):
            cls.save_checkpoint_dir = Path(kwargs['save_checkpoint_dir'])
        if kwargs.get('load_checkpoint_dir'):
            cls.load_checkpoint_dir = Path(kwargs['load_checkpoint_dir'])
        
        # Device - store raw value, resolve later
        device = kwargs.get('device', 'auto')
        cls._device_raw = device
        cls.device = device  # Will be resolved when needed
        
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
        Sync adaptive yield using thread-safe implementation.
        The old loop._run_once() approach never worked and has been removed.
        """
        if cls._yield_disabled > 0 or cls.no_yield:
            return
        
        start_time = time.perf_counter()
        cls._concurrency_stats.start_yield()
        
        # Use thread-safe yielding
        thread_safe_sync_adaptive_yield(delay=cls._compute_adaptive_delay())
        
        # Update statistics
        end_time = time.perf_counter()
        yield_duration = end_time - start_time
        cls._yield_stats.update(yield_duration)
        cls._concurrency_stats.end_yield(yield_duration)
    
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
    def get_concurrency_stats(cls) -> Dict[str, Any]:
        """
        Get statistics about concurrent execution balance.
        
        Returns:
            Dictionary with concurrency statistics
        """
        ppo_pct, non_ppo_pct = cls._concurrency_stats.get_balance_percentage()
        return {
            'ppo_time_s': cls._concurrency_stats.ppo_time,
            'non_ppo_time_s': cls._concurrency_stats.non_ppo_time,
            'ppo_percentage': ppo_pct,
            'non_ppo_percentage': non_ppo_pct,
            'total_execution_time_s': cls._concurrency_stats.ppo_time + cls._concurrency_stats.non_ppo_time
        }
    
    @classmethod
    def print_concurrency_report(cls):
        """Print a formatted report of concurrent execution balance"""
        stats = cls.get_concurrency_stats()
        
        print("\n" + "="*60)
        print("🔄 CONCURRENT EXECUTION BALANCE REPORT")
        print("="*60)
        print(f"Total execution time: {stats['total_execution_time_s']:.2f}s")
        print(f"PPO subgraph time:    {stats['ppo_time_s']:.2f}s ({stats['ppo_percentage']:.1f}%)")
        print(f"MNIST subgraph time:  {stats['non_ppo_time_s']:.2f}s ({stats['non_ppo_percentage']:.1f}%)")
        print(f"Total yields:         {cls._yield_stats.total_yields}")
        
        if stats['ppo_percentage'] > 0 and stats['non_ppo_percentage'] > 0:
            print("\n✅ Both subgraphs are receiving execution time!")
            if abs(stats['ppo_percentage'] - stats['non_ppo_percentage']) < 20:
                print("   Execution is well-balanced between subgraphs.")
            elif stats['ppo_percentage'] > stats['non_ppo_percentage']:
                print("   PPO subgraph is dominating execution time.")
            else:
                print("   MNIST subgraph is dominating execution time.")
        elif stats['ppo_percentage'] == 0:
            print("\n⚠️  No PPO execution detected - sync_adaptive_yield may not be called")
        elif stats['non_ppo_percentage'] == 0:
            print("\n⚠️  No MNIST execution detected - async yields may be blocked")
        
        print("="*60)
        
        # Force flush to ensure output is visible
        import sys
        sys.stdout.flush()
    
    @classmethod
    def reset_metrics(cls):
        """Reset all performance metrics (useful for benchmarking)"""
        cls._yield_stats = YieldStats()
        cls._concurrency_stats = ConcurrencyStats()
        cls._node_metrics.clear()
        cls._total_queued = 0
    
    @classmethod
    def get_device(cls) -> str:
        """
        Get the device, resolving 'auto' if needed.
        This is called lazily to avoid importing torch too early.
        """
        if cls._device_raw == 'auto' and cls.device == 'auto':
            # Resolve auto device now
            try:
                import torch
                cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                if cls.verbose:
                    cls._logger.info(f"Resolved device: {cls.device}")
            except ImportError:
                # If torch not available, default to cpu
                cls.device = 'cpu'
                if cls.verbose:
                    cls._logger.warning("PyTorch not available, defaulting to cpu")
        return cls.device
    
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
    
    @classmethod
    def get_node_config(cls, node_id: str, key: str, default: Any = None) -> Any:
        """
        Get configuration value for a specific node.
        
        Args:
            node_id: Unique identifier of the node
            key: Configuration key to retrieve
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return cls.node_configs.get(node_id, {}).get(key, default)
    
    @classmethod
    def set_node_config(cls, node_id: str, key: str, value: Any):
        """
        Set configuration value for a specific node.
        
        Args:
            node_id: Unique identifier of the node
            key: Configuration key to set
            value: Configuration value
        """
        if node_id not in cls.node_configs:
            cls.node_configs[node_id] = {}
        cls.node_configs[node_id][key] = value
        
        if cls.verbose:
            cls._logger.info(f"Set node config: {node_id}.{key} = {value}")


# All configuration should now be accessed through Global class
# Example: from framework.globals import Global as g
#          if g.verbose: ...