#!/usr/bin/env python3
"""
Simple test to reproduce the yield scheduling issue.
Three tasks: A and B communicate via queues, C does sync yielding.
"""

import asyncio
import time
import sys
import argparse


class SimpleSyncYielder:
    """Simplified version of sync_adaptive_yield"""
    
    @classmethod
    def sync_yield(cls, delay: float = 0.05):
        """Sync yield using event loop internals"""
        start_time = time.perf_counter()
        
        # Get event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            print("[YIELD] No running loop!")
            return
            
        # Print queue state before yield
        if hasattr(loop, '_ready'):
            print(f"[YIELD] Ready queue before: {len(loop._ready)} items")
        
        if delay == 0:
            # Quick yield
            loop.call_soon(lambda: None)
            loop._run_once()
        else:
            # Timed delay
            done = False
            
            def set_done():
                nonlocal done
                done = True
            
            # Schedule callback after delay
            loop.call_later(delay, set_done)
            
            # Run event loop until timer fires
            run_count = 0
            while not done:
                run_count += 1
                if run_count <= 3:
                    print(f"[YIELD] While loop iteration {run_count}, done={done}")
                loop._run_once()
        
        end_time = time.perf_counter()
        yield_duration = end_time - start_time
        print(f"[YIELD] Completed in {yield_duration*1000:.1f}ms")


async def task_a(queue_a_to_b: asyncio.Queue, queue_b_to_a: asyncio.Queue):
    """Task A: Sends to B, waits for response"""
    print("[A] Starting")
    
    for i in range(30):
        # Send message to B
        msg = f"Message {i} from A"
        print(f"[A] Sending: {msg}")
        await queue_a_to_b.put(msg)
        
        # Wait for response from B
        print(f"[A] Waiting for response...")
        response = await queue_b_to_a.get()
        print(f"[A] Received: {response}")
        
        # Wait 1 second before next message
        await asyncio.sleep(1.0)
    
    print("[A] Completed 30 exchanges")
    await queue_a_to_b.put("DONE")


async def task_b(queue_a_to_b: asyncio.Queue, queue_b_to_a: asyncio.Queue):
    """Task B: Receives from A, sends response"""
    print("[B] Starting")
    
    while True:
        # Wait for message from A
        print(f"[B] Waiting for message...")
        msg = await queue_a_to_b.get()
        print(f"[B] Received: {msg}")
        
        if msg == "DONE":
            break
            
        # Wait 1 second
        await asyncio.sleep(1.0)
        
        # Send response to A
        response = f"Response to '{msg}' from B"
        print(f"[B] Sending: {response}")
        await queue_b_to_a.put(response)
    
    print("[B] Completed")


async def task_c_yielder():
    """Task C: Continuously calls sync_yield"""
    print("[C] Starting yielder task")
    
    # First, let's try calling sync_yield from a synchronous context
    # by using run_in_executor
    loop = asyncio.get_running_loop()
    
    def sync_yield_wrapper():
        """Run sync_yield in a thread"""
        # This won't work because the event loop is not accessible from threads
        SimpleSyncYielder.sync_yield(delay=0.5)
    
    for i in range(100):  # Run for a while
        print(f"\n[C] Yield #{i}")
        
        # Try 1: Direct call (will cause the error we saw)
        # SimpleSyncYielder.sync_yield(delay=0.5)
        
        # Try 2: Just use regular async sleep
        await asyncio.sleep(0.5)
        
        # This allows the event loop to run other tasks naturally
    
    print("[C] Yielder completed")


async def main(test_yield: bool):
    """Main async function"""
    print("=== Starting Simple Yield Test ===")
    print(f"Test yield: {test_yield}")
    
    # Create queues
    queue_a_to_b = asyncio.Queue(maxsize=10)
    queue_b_to_a = asyncio.Queue(maxsize=10)
    
    # Create tasks
    tasks = [
        asyncio.create_task(task_a(queue_a_to_b, queue_b_to_a), name="TaskA"),
        asyncio.create_task(task_b(queue_a_to_b, queue_b_to_a), name="TaskB")
    ]
    
    if test_yield:
        tasks.append(asyncio.create_task(task_c_yielder(), name="TaskC"))
    
    # Wait for all tasks to complete
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Error: {e}")
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
    
    print("=== Test Completed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple yield test")
    parser.add_argument("-test-yield", action="store_true", 
                        help="Enable sync yield testing with task C")
    args = parser.parse_args()
    
    # Run the async main
    asyncio.run(main(args.test_yield))