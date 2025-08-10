#!/usr/bin/env python3
"""
Test using run_in_executor to run sync code with yielding.
This explores the standard asyncio approach for running sync code.
"""

import asyncio
import time
import concurrent.futures
import argparse


async def task_a(queue_a_to_b: asyncio.Queue, queue_b_to_a: asyncio.Queue):
    """Task A: Sends to B, waits for response"""
    print("[A] Starting")
    
    for i in range(30):
        msg = f"Message {i} from A"
        print(f"[A] Sending: {msg}")
        await queue_a_to_b.put(msg)
        
        print(f"[A] Waiting for response...")
        response = await queue_b_to_a.get()
        print(f"[A] Received: {response}")
        
        await asyncio.sleep(1.0)
    
    print("[A] Completed 30 exchanges")
    await queue_a_to_b.put("DONE")


async def task_b(queue_a_to_b: asyncio.Queue, queue_b_to_a: asyncio.Queue):
    """Task B: Receives from A, sends response"""
    print("[B] Starting")
    
    while True:
        print(f"[B] Waiting for message...")
        msg = await queue_a_to_b.get()
        print(f"[B] Received: {msg}")
        
        if msg == "DONE":
            break
            
        await asyncio.sleep(1.0)
        
        response = f"Response to '{msg}' from B"
        print(f"[B] Sending: {response}")
        await queue_b_to_a.put(response)
    
    print("[B] Completed")


def sync_training_step(step: int):
    """Single synchronous training step"""
    print(f"[EXECUTOR] Training step {step}")
    time.sleep(0.1)  # Simulate work
    return f"Step {step} completed"


async def task_c_executor():
    """Task C: Uses run_in_executor for sync code"""
    print("[C] Starting executor task")
    
    # Get event loop and executor
    loop = asyncio.get_running_loop()
    
    # Use ThreadPoolExecutor for sync code
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for i in range(30):
            # Run sync code in executor
            result = await loop.run_in_executor(executor, sync_training_step, i)
            print(f"[C] {result}")
            
            # Yield every 5 steps by using async sleep
            if i > 0 and i % 5 == 0:
                print(f"[C] Yielding control...")
                await asyncio.sleep(0.5)
    
    print("[C] Executor completed")


async def main(test_yield: bool):
    """Main async function"""
    print("=== Starting Executor Yield Test ===")
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
        tasks.append(asyncio.create_task(task_c_executor(), name="TaskC"))
    
    # Wait for all tasks to complete
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Error: {e}")
        for task in tasks:
            if not task.done():
                task.cancel()
    
    print("=== Test Completed ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executor yield test")
    parser.add_argument("-test-yield", action="store_true", 
                        help="Enable executor yield testing with task C")
    args = parser.parse_args()
    
    asyncio.run(main(args.test_yield))