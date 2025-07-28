#!/usr/bin/env python3
"""
Metrics integration for runner.py
Add this to runner.py to enable metrics logging
"""

# Add to imports section:
# from framework.metrics_logger import get_metrics_logger

# Add after GraphRunner initialization:
def setup_metrics_logging():
    """Setup metrics logging for the workflow"""
    try:
        from framework.metrics_logger import get_metrics_logger
        logger = get_metrics_logger()
        
        # Register shutdown handler
        import atexit
        atexit.register(lambda: logger.shutdown())
        
        # Periodic summary generation
        import asyncio
        async def periodic_summary():
            while True:
                await asyncio.sleep(30)  # Generate summary every 30 seconds
                logger.generate_summary()
        
        # Start periodic summary task
        asyncio.create_task(periodic_summary())
        
        print("📊 Metrics logging enabled - logs will be saved to metrics_logs/")
        return logger
    except Exception as e:
        print(f"⚠️  Metrics logging not available: {e}")
        return None

# Add this after creating GraphRunner:
# metrics_logger = setup_metrics_logging()

# Example usage in nodes:
"""
# In any node's compute method:
try:
    from framework.metrics_logger import get_metrics_logger
    logger = get_metrics_logger()
    
    # Record metrics
    logger.record_metric(self.node_id, "MyNode", "processing_time", elapsed_ms)
    logger.record_metric(self.node_id, "MyNode", "batch_size", batch.shape[0])
    
    # Record violations
    if error_rate > threshold:
        logger.record_violation(self.node_id, "MyNode", "error_rate_exceeded", 
                               threshold, error_rate)
except ImportError:
    # Metrics logger not available
    pass
"""