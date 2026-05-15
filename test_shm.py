#!/usr/bin/env python3
import multiprocessing
from multiprocessing import shared_memory
import time
import sys

def child(name):
    try:
        shm = shared_memory.SharedMemory(name=name)
        print(f"Child: opened shm '{name}', size={shm.size}", flush=True)
        shm.buf[0] = 42
        print(f"Child: wrote 42 to shm", flush=True)
        shm.close()
    except Exception as e:
        print(f"Child: FAILED to open shm '{name}': {e}", flush=True)

def main():
    shm = shared_memory.SharedMemory(create=True, size=1024)
    print(f"Parent: created shm '{shm.name}', size={shm.size}", flush=True)

    p = multiprocessing.Process(target=child, args=(shm.name,))
    p.start()
    p.join()

    val = shm.buf[0]
    print(f"Parent: read {val} from shm (expected 42)", flush=True)

    shm.close()
    shm.unlink()

    if val == 42:
        print("SUCCESS: shared memory works between parent and child", flush=True)
    else:
        print("FAILURE: shared memory NOT working", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
