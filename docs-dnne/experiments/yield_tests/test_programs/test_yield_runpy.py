#!/usr/bin/env python3
"""
Test that simulates PPO's use of runpy.run_path to execute synchronous code
that needs to yield to allow async tasks to run.
"""

import asyncio
import time
import sys
import argparse
import runpy
import os


# Create a separate Python file to be executed via runpy
SYNC_SCRIPT = """
# sync_code.py - Simulates PPO training loop
import time
import asyncio

def sync_yield(delay=0.05):
    '''Yield from synchronous code'''
    print(f"[SYNC] sync_yield called with delay={delay}")
    
    # Get the loop passed from parent
    import sys
    if hasattr(sys.modules['__main__'], 'dnne_loop'):
        loop = sys.modules['__main__'].dnne_loop
        print(f"[SYNC] Got loop from parent: {loop}")
    else:
        try:
            loop = asyncio.get_running_loop()
            print(f"[SYNC] Got running loop: {loop}")
        except RuntimeError:
            print("[SYNC] No running loop!")
            return
    
    # Try to yield
    done = False
    def set_done():
        nonlocal done
        done = True
    
    loop.call_later(delay, set_done)
    
    # This is the problematic part - calling _run_once from sync code
    while not done:
        loop._run_once()
    
    print(f"[SYNC] sync_yield completed")

# Main sync training loop
print("[SYNC] Starting synchronous training loop")
for i in range(30):
    print(f"[SYNC] Training step {i}")
    time.sleep(0.1)  # Simulate work
    
    # Try to yield every 5 steps
    if i % 5 == 0:
        sync_yield(0.5)

print("[SYNC] Training completed")
"""


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


async def task_c_runpy():
    """Task C: Uses runpy to execute sync code (simulates PPO)"""
    print("[C] Starting runpy task")
    
    # Write the sync script to a file
    with open("sync_code.py", "w") as f:
        f.write(SYNC_SCRIPT)
    
    # Get current event loop to pass to sync code
    loop = asyncio.get_running_loop()
    print(f"[C] Passing loop to sync code: {loop}")
    
    # Run sync code via runpy (like PPO does)
    try:
        # This simulates how PPO runs training
        result = runpy.run_path("sync_code.py", 
                               init_globals={"dnne_loop": loop}, 
                               run_name="__main__")
    except Exception as e:
        print(f"[C] Error in runpy: {e}")
    
    print("[C] Runpy completed")
    
    # Clean up
    os.remove("sync_code.py")


async def main(test_yield: bool):
    """Main async function"""
    print("=== Starting Runpy Yield Test ===")
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
        tasks.append(asyncio.create_task(task_c_runpy(), name="TaskC"))
    
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
    parser = argparse.ArgumentParser(description="Runpy yield test")
    parser.add_argument("-test-yield", action="store_true", 
                        help="Enable runpy yield testing with task C")
    args = parser.parse_args()
    
    asyncio.run(main(args.test_yield))