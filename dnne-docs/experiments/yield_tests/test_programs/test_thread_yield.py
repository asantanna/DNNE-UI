#!/usr/bin/env python3
"""
Test using threading to run sync code that needs to yield to async tasks.
This explores running PPO in a separate thread to avoid the task context issue.
"""

import asyncio
import time
import threading
import queue
import argparse


class ThreadedYielder:
    """Manages yielding from sync code running in a thread"""
    
    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop
        self.yield_queue = queue.Queue()  # Thread-safe queue for communication
        
    def sync_yield(self, delay: float = 0.05):
        """Called from thread to yield control"""
        print(f"[THREAD] Requesting yield with delay={delay}")
        
        # Create a future in the main loop
        future = asyncio.Future()
        
        # Schedule the yield in the event loop
        def do_yield():
            async def yield_coro():
                await asyncio.sleep(delay)
                future.set_result(True)
            
            # Create task in the event loop
            asyncio.create_task(yield_coro())
        
        # Schedule in the event loop from thread
        self.loop.call_soon_threadsafe(do_yield)
        
        # Wait for completion (blocks thread but not event loop)
        start = time.time()
        while not future.done():
            time.sleep(0.001)  # Small sleep to avoid busy waiting
            if time.time() - start > delay + 1.0:  # Timeout
                print("[THREAD] Yield timeout!")
                break
        
        print(f"[THREAD] Yield completed")


def sync_training_loop(yielder: ThreadedYielder):
    """Simulates PPO training running in a thread"""
    print("[THREAD] Starting synchronous training loop")
    
    for i in range(30):
        print(f"[THREAD] Training step {i}")
        time.sleep(0.1)  # Simulate work
        
        # Yield every 5 steps
        if i > 0 and i % 5 == 0:
            yielder.sync_yield(0.5)
    
    print("[THREAD] Training completed")


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


async def task_c_thread(loop: asyncio.AbstractEventLoop):
    """Task C: Runs sync code in a thread"""
    print("[C] Starting thread task")
    
    # Create yielder for thread communication
    yielder = ThreadedYielder(loop)
    
    # Run sync training in a thread
    thread = threading.Thread(target=sync_training_loop, args=(yielder,))
    thread.start()
    
    # Wait for thread to complete
    await asyncio.get_event_loop().run_in_executor(None, thread.join)
    
    print("[C] Thread completed")


async def main(test_yield: bool):
    """Main async function"""
    print("=== Starting Thread Yield Test ===")
    print(f"Test yield: {test_yield}")
    
    # Create queues
    queue_a_to_b = asyncio.Queue(maxsize=10)
    queue_b_to_a = asyncio.Queue(maxsize=10)
    
    # Get event loop for thread communication
    loop = asyncio.get_running_loop()
    
    # Create tasks
    tasks = [
        asyncio.create_task(task_a(queue_a_to_b, queue_b_to_a), name="TaskA"),
        asyncio.create_task(task_b(queue_a_to_b, queue_b_to_a), name="TaskB")
    ]
    
    if test_yield:
        tasks.append(asyncio.create_task(task_c_thread(loop), name="TaskC"))
    
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
    parser = argparse.ArgumentParser(description="Thread yield test")
    parser.add_argument("-test-yield", action="store_true", 
                        help="Enable thread yield testing with task C")
    args = parser.parse_args()
    
    asyncio.run(main(args.test_yield))