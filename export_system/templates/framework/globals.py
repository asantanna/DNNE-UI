"""
Global state management for DNNE exported workflows.
Provides centralized configuration and adaptive yielding.
"""

# Set up warning filters before any other imports
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*invalid escape sequence.*')
warnings.filterwarnings('ignore', message='.*Using or importing the ABCs from.*')
warnings.filterwarnings('ignore', message='.*np.int.*is a deprecated alias.*')
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')
warnings.filterwarnings('ignore', message='.*declare_namespace.*')

import asyncio
import time
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from pathlib import Path

# Thread-safe yielding for sync code
from .globals_threadsafe import thread_safe_sync_adaptive_yield


class DNNE_Logging:
    """Wrapper module that prepends 'dnne.' to logger names"""
    
    def getLogger(self, name=""):
        """Get a DNNE logger with automatic prefix"""
        if name == "":
            return logging.getLogger("dnne")
        elif name.startswith("dnne."):
            # Already has prefix, don't double it
            return logging.getLogger(name)
        else:
            # Add prefix
            return logging.getLogger(f"dnne.{name}")
    
    def __getattr__(self, name):
        """Pass through all other logging attributes unchanged"""
        return getattr(logging, name)


# Create module-like instance for easy importing
dnne_logging = DNNE_Logging()

# Create yield logger for adaptive yielding subsystem
yield_logger = dnne_logging.getLogger("yield")

# Create balancing logger for execution balance reports
balancing_logger = dnne_logging.getLogger("balancing")


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
class SubgraphMetrics:
    """Metrics for a computational subgraph"""
    subgraph_name: str
    node_type: str = "async"  # "sync" or "async"
    item_unit: str = "items"  # e.g., "env_steps", "batches(64)"
    
    # Common metrics
    items_processed: int = 0
    last_item_time: float = 0.0
    start_time: float = field(default_factory=time.time)
    
    # Sync-only metrics
    cpu_time: float = 0.0  # Total CPU seconds used
    last_yield_time: float = 0.0  # When we last yielded
    
    @property
    def throughput(self) -> float:
        """Items per second"""
        if self.last_item_time == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        return self.items_processed / elapsed if elapsed > 0 else 0.0
    
    @property
    def cpu_percentage(self) -> Optional[float]:
        """CPU percentage (sync nodes only)"""
        if self.node_type != "sync":
            return None
        elapsed = time.time() - self.start_time
        return (self.cpu_time / elapsed * 100) if elapsed > 0 else 0.0
    


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
    _concurrency_stats: ConcurrencyStats = ConcurrencyStats()  # Keep for backward compatibility
    _subgraph_metrics: Dict[str, SubgraphMetrics] = {}  # subgraph_name -> metrics
    _node_to_subgraph: Dict[str, str] = {}  # node_id -> subgraph_name
    _total_queued: int = 0
    _initialized: bool = False
    _logger: Optional[logging.Logger] = None
    _start_time: float = 0.0  # Workflow start time
    
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
        # Store raw values for subsystem support
        verbose_arg = kwargs.get('verbose', False)
        debug_arg = kwargs.get('debug', False)
        
        # Set boolean flags based on whether any verbosity is enabled
        cls.verbose = bool(verbose_arg)
        cls.profiling = kwargs.get('profiling', False)
        cls.debug = bool(debug_arg)
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
        cls._start_time = time.time()
        
        # Log initialization
        if cls.verbose:
            cls._logger.info(f"Global initialized: mode={'inference' if cls.inference_mode else 'training'}, "
                           f"device={cls.device}, verbose={cls.verbose}")
        
        if cls.no_yield:
            cls._logger.info("Adaptive yielding DISABLED - running at full speed for performance comparison")
    
    @classmethod
    async def async_adaptive_yield(cls, *, subgraph: str, is_item_ref: bool = False):
        """
        Asynchronous adaptive yield for async nodes.
        
        Args:
            subgraph: Name of the subgraph (e.g., "mnist")
            is_item_ref: True if this yield represents one work item completion
        """
        
        # Fast path - no yield if disabled by context or command line
        if cls._yield_disabled > 0 or cls.no_yield:
            return
            
        start_time = time.perf_counter()
        
        # Compute adaptive delay
        delay = cls._compute_adaptive_delay()
        
        # Perform yield
        await asyncio.sleep(delay)
        
        # Update statistics and track items for async nodes
        end_time = time.perf_counter()
        yield_duration = end_time - start_time
        cls._yield_stats.update(yield_duration)
        
        # Track items for async nodes
        if is_item_ref and subgraph in cls._subgraph_metrics:
            metrics = cls._subgraph_metrics[subgraph]
            metrics.items_processed += 1
            metrics.last_item_time = time.time()
        
        # Log if excessive yield time
        if cls.debug and yield_duration > 0.01:  # 10ms
            yield_logger.warning(f"Long yield detected: {yield_duration*1000:.2f}ms from subgraph={subgraph}")
    
    @classmethod
    def sync_adaptive_yield(cls, *, subgraph: str, is_item_ref: bool = False):
        """
        Synchronous adaptive yield for thread-based nodes.
        
        Args:
            subgraph: Name of the subgraph (e.g., "ppo")
            is_item_ref: True if this yield represents one work item completion
        """
        if cls._yield_disabled > 0 or cls.no_yield:
            return
        
        start_time = time.perf_counter()
        
        # Track CPU time and items for the subgraph
        if subgraph in cls._subgraph_metrics:
            metrics = cls._subgraph_metrics[subgraph]
            if metrics.last_yield_time > 0:
                cpu_duration = start_time - metrics.last_yield_time
                metrics.cpu_time += cpu_duration
            if is_item_ref:
                metrics.items_processed += 1
                metrics.last_item_time = time.time()
        
        # Keep backward compatibility with old concurrency stats
        cls._concurrency_stats.start_yield()
        
        # Log yield start in debug mode
        if cls.debug:
            yield_logger.debug(f"Starting sync adaptive yield from subgraph={subgraph}")
        
        # Use thread-safe yielding
        delay = cls._compute_adaptive_delay()
        thread_safe_sync_adaptive_yield(delay=delay)
        
        # Update statistics
        end_time = time.perf_counter()
        yield_duration = end_time - start_time
        cls._yield_stats.update(yield_duration)
        cls._concurrency_stats.end_yield(yield_duration)
        
        # Update subgraph yield time
        if subgraph in cls._subgraph_metrics:
            cls._subgraph_metrics[subgraph].last_yield_time = end_time
        
        # Log yield completion in debug mode
        if cls.debug:
            yield_logger.debug(f"Yield completed in {yield_duration*1000:.2f}ms with delay {delay*1000:.2f}ms from subgraph={subgraph}")
    
    @classmethod
    def _compute_adaptive_delay(cls) -> float:
        """
        Compute appropriate yield delay based on system metrics.
        
        Returns:
            Delay in seconds (0 for minimal yield, up to 0.01 for aggressive yielding)
        """
        # For now, always use minimal yield
        # TODO: Implement adaptive delay based on subgraph requirements
        return 0.0
    
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
            'subgraphs_tracked': len(cls._subgraph_metrics)
        }
    
    @classmethod
    def register_sync_node(cls, node_id: str, subgraph: str, item_unit: str = "items", requirements: Optional[Dict[str, Any]] = None):
        """
        Register a synchronous node (like PPO) with the balancer.
        
        Args:
            node_id: Unique identifier for the node
            subgraph: Name of the subgraph this node belongs to
            item_unit: Unit name for items processed (e.g., "env_steps")
            requirements: Optional balancing requirements from config
        """
        cls._node_to_subgraph[node_id] = subgraph
        
        if subgraph not in cls._subgraph_metrics:
            cls._subgraph_metrics[subgraph] = SubgraphMetrics(
                subgraph_name=subgraph,
                node_type="sync",
                item_unit=item_unit
            )
            balancing_logger.info(f"Registered sync node {node_id} in subgraph '{subgraph}' with unit '{item_unit}'")
    
    @classmethod
    def register_balancing_node(cls, node_id: str, subgraph: str, item_unit: str = "items", requirements: Optional[Dict[str, Any]] = None):
        """
        Register a balancing node (async) with the balancer.
        
        Args:
            node_id: Unique identifier for the node
            subgraph: Name of the subgraph this node belongs to
            item_unit: Unit name for items processed (e.g., "batches(64)")
            requirements: Optional balancing requirements from config
        """
        cls._node_to_subgraph[node_id] = subgraph
        
        if subgraph not in cls._subgraph_metrics:
            cls._subgraph_metrics[subgraph] = SubgraphMetrics(
                subgraph_name=subgraph,
                node_type="async",
                item_unit=item_unit
            )
            balancing_logger.info(f"Registered balancing node {node_id} in subgraph '{subgraph}' with unit '{item_unit}'")
    
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
    def print_concurrency_report(cls, force_print=False):
        """Print a formatted report of concurrent execution balance
        
        Args:
            force_print: If True, always print regardless of logging level (for final report)
        """
        # Calculate totals
        total_time = time.time() - cls._start_time if cls._start_time > 0 else 0
        total_sync_cpu = sum(m.cpu_time for m in cls._subgraph_metrics.values() if m.node_type == "sync")
        total_async_time = total_time - total_sync_cpu if total_time > 0 else 0
        
        # Construct the report as a multi-line string
        report_lines = []
        report_lines.append("\n" + "="*60)
        report_lines.append("🔄 EXECUTION BALANCE REPORT")
        report_lines.append("="*60)
        report_lines.append(f"Total execution time: {total_time:.2f}s")
        report_lines.append(f"Sync nodes CPU time:  {total_sync_cpu:.2f}s ({total_sync_cpu/total_time*100:.1f}%)" if total_time > 0 else "Sync nodes CPU time:  0.00s (0.0%)")
        report_lines.append(f"Async nodes time:     {total_async_time:.2f}s ({total_async_time/total_time*100:.1f}%)" if total_time > 0 else "Async nodes time:     0.00s (0.0%)")
        
        # Show subgraph performance
        if cls._subgraph_metrics:
            report_lines.append("\nSubgraph Performance:")
            for name, metrics in sorted(cls._subgraph_metrics.items()):
                if metrics.node_type == "sync":
                    cpu_pct = metrics.cpu_percentage
                    report_lines.append(f"  {name:8s}: {metrics.throughput:6.1f} {metrics.item_unit}/sec ({cpu_pct:.1f}% CPU)")
                else:
                    report_lines.append(f"  {name:8s}: {metrics.throughput:6.1f} {metrics.item_unit}/sec (async - CPU % N/A)")
        
        # Also show legacy PPO vs non-PPO stats if available (for backward compatibility)
        if cls._concurrency_stats.ppo_time > 0 or cls._concurrency_stats.non_ppo_time > 0:
            ppo_pct, non_ppo_pct = cls._concurrency_stats.get_balance_percentage()
            report_lines.append("\nLegacy Stats (PPO vs yield time):")
            report_lines.append(f"  PPO CPU time:  {cls._concurrency_stats.ppo_time:.2f}s ({ppo_pct:.1f}%)")
            report_lines.append(f"  Yield time:    {cls._concurrency_stats.non_ppo_time:.2f}s ({non_ppo_pct:.1f}%)")
        
        report_lines.append("\nNote: Async time includes all async activity (MNIST,")
        report_lines.append("      system overhead, idle time)")
        report_lines.append("="*60)
        
        # Join the report
        report = "\n".join(report_lines)
        
        # For final report (from runner.py), always print
        if force_print:
            print(report)
        else:
            # For periodic reports, use the balancing logger at debug level
            balancing_logger.debug(report)
        
        # Force flush to ensure output is visible
        import sys
        sys.stdout.flush()
    
    @classmethod
    def reset_metrics(cls):
        """Reset all performance metrics (useful for benchmarking)"""
        cls._yield_stats = YieldStats()
        cls._concurrency_stats = ConcurrencyStats()
        cls._subgraph_metrics.clear()
        cls._node_to_subgraph.clear()
        cls._total_queued = 0
        cls._start_time = time.time()
    
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
            True if system is healthy (placeholder for future implementation)
        """
        # TODO: Implement health check based on subgraph requirements
        return True
    
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