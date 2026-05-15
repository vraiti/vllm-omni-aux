#!/usr/bin/env python3
"""Test shared memory with mp.spawn (matching vllm-omni's start method)."""
import multiprocessing as mp
import time
import sys
from multiprocessing import shared_memory

def child_create_shm(pipe_w, name_from_parent):
    # Open parent's shm
    try:
        parent_shm = shared_memory.SharedMemory(name=name_from_parent)
        print(f"Child: opened parent shm '{name_from_parent}', val={parent_shm.buf[0]}", flush=True)
        parent_shm.close()
    except Exception as e:
        print(f"Child: FAILED to open parent shm: {e}", flush=True)

    # Create child shm
    child_shm = shared_memory.SharedMemory(create=True, size=1024)
    child_shm.buf[0] = 99
    print(f"Child: created shm '{child_shm.name}'", flush=True)

    # Send name back via pipe
    pipe_w.send(child_shm.name)
    pipe_w.close()

    # Keep alive long enough for parent to read
    time.sleep(5)
    child_shm.close()
    child_shm.unlink()

def main():
    mp.set_start_method("spawn", force=True)

    # Parent creates shm
    parent_shm = shared_memory.SharedMemory(create=True, size=1024)
    parent_shm.buf[0] = 77
    print(f"Parent: created shm '{parent_shm.name}'", flush=True)

    r, w = mp.Pipe(duplex=False)
    p = mp.Process(target=child_create_shm, args=(w, parent_shm.name))
    p.start()
    w.close()

    child_shm_name = r.recv()
    r.close()
    print(f"Parent: received child shm name '{child_shm_name}'", flush=True)

    try:
        child_shm = shared_memory.SharedMemory(name=child_shm_name)
        val = child_shm.buf[0]
        print(f"Parent: opened child shm, val={val} (expected 99)", flush=True)
        child_shm.close()
        if val == 99:
            print("SUCCESS: spawn + bidirectional shm works", flush=True)
        else:
            print("FAILURE: wrong value", flush=True)
    except FileNotFoundError as e:
        print(f"FAILURE: parent cannot open child shm: {e}", flush=True)
        sys.exit(1)

    p.join()
    parent_shm.close()
    parent_shm.unlink()

if __name__ == "__main__":
    main()
