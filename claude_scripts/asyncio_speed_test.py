#!/usr/bin/env python3
"""
AsyncIO Ping-Pong Speed Test
Tests maximum speed of asyncio queue exchanges to measure overhead
"""

import asyncio
import time

async def task_a(queue_a_to_b, queue_b_to_a, duration=30):
    """Task A: Send data and wait for response"""
    counter = 0
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Send counter to Task B
        await queue_a_to_b.put(counter)
        
        # Wait for response from Task B
        response = await queue_b_to_a.get()
        
        counter += 1
    
    # Send termination signal
    await queue_a_to_b.put(None)
    
    return counter

async def task_b(queue_a_to_b, queue_b_to_a):
    """Task B: Relay data immediately back to Task A"""
    while True:
        # Get data from Task A
        data = await queue_a_to_b.get()
        
        # Check for termination
        if data is None:
            break
            
        # Immediately send it back (no processing)
        await queue_b_to_a.put(data)

async def run_ping_pong_test(duration=30):
    """Run the ping-pong test for specified duration"""
    print(f"Starting AsyncIO ping-pong test for {duration} seconds...")
    print("Task A sends data → Task B immediately relays back")
    print("Queue size = 1 (maximum back-pressure)")
    print()
    
    # Create queues with size 1 for maximum back-pressure
    queue_a_to_b = asyncio.Queue(maxsize=1)
    queue_b_to_a = asyncio.Queue(maxsize=1)
    
    # Record start time
    start_time = time.time()
    
    # Start both tasks
    task_a_coro = task_a(queue_a_to_b, queue_b_to_a, duration)
    task_b_coro = task_b(queue_a_to_b, queue_b_to_a)
    
    # Run both tasks concurrently
    results = await asyncio.gather(task_a_coro, task_b_coro)
    
    # Record end time
    end_time = time.time()
    actual_duration = end_time - start_time
    
    # Get results
    total_exchanges = results[0]  # Task A returns the counter
    
    # Calculate performance metrics
    exchanges_per_second = total_exchanges / actual_duration
    microseconds_per_exchange = (actual_duration * 1_000_000) / total_exchanges
    
    print("=" * 60)
    print("ASYNCIO PING-PONG PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"Duration: {actual_duration:.2f} seconds")
    print(f"Total round-trip exchanges: {total_exchanges:,}")
    print(f"Exchanges per second: {exchanges_per_second:,.0f}")
    print(f"Microseconds per exchange: {microseconds_per_exchange:.1f} μs")
    print()
    
    # Compare with theoretical maximum
    theoretical_max = 1_000_000 / microseconds_per_exchange
    print(f"Theoretical maximum (1 CPU): {theoretical_max:,.0f} exchanges/sec")
    
    # Estimate overhead
    if microseconds_per_exchange < 1000:  # Less than 1ms
        print("✅ AsyncIO overhead appears minimal (<1ms per exchange)")
    else:
        print("⚠️  AsyncIO overhead is significant (>1ms per exchange)")
    
    return exchanges_per_second, microseconds_per_exchange

if __name__ == "__main__":
    # Run the test
    asyncio.run(run_ping_pong_test(duration=30))