#!/usr/bin/env python3
"""
Test converting PPO to async code with proper yielding.
This explores restructuring sync code to use async/await.
"""

import asyncio
import time
import argparse


async def async_training_loop():
    """Async version of PPO training with natural yielding"""
    print("[ASYNC_PPO] Starting asynchronous training loop")
    
    for i in range(30):
        print(f"[ASYNC_PPO] Training step {i}")
        
        # Simulate work with async sleep (naturally yields)
        await asyncio.sleep(0.1)
        
        # Additional yield every 5 steps
        if i > 0 and i % 5 == 0:
            print(f"[ASYNC_PPO] Extra yield at step {i}")
            await asyncio.sleep(0.5)
    
    print("[ASYNC_PPO] Training completed")


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


async def main(test_yield: bool):
    """Main async function"""
    print("=== Starting Async PPO Test ===")
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
        tasks.append(asyncio.create_task(async_training_loop(), name="TaskC_Async"))
    
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
    parser = argparse.ArgumentParser(description="Async PPO test")
    parser.add_argument("-test-yield", action="store_true", 
                        help="Enable async PPO testing with task C")
    args = parser.parse_args()
    
    asyncio.run(main(args.test_yield))