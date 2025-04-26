import time
import psutil
import os
import gc
def _check_dims(*args):
    """Check that all args have the same shape along the first dimension."""
    if len(args) == 0:
        return
    shape = args[0].shape
    for arg in args[1:]:
        if arg.shape != shape:
            raise ValueError(
                "Found arguments with inconsistent shapes: {}".format(
                    ", ".join(map(str, args))
                )
            )

def check_dims_decorator(*pos):
    def decorator(func):
        def wrapper(*args,**kargs):
            _check_dims(*[args[i] for i in pos])
            return func(*args,**kargs)
        return wrapper
    return decorator

def time_decorator(func):
    def wrapper(*args,**kargs):
        start = time.time()
        result = func(*args,**kargs)
        end = time.time()
        print("Time taken for {} is {} seconds".format(func.__name__,end-start))
        return result
    return wrapper

def mem_usage(func):
    
    def wrapper(*args,**kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        result = func(*args,**kwargs)
        mem_after = process.memory_info().rss
        print("Memory usage for {} is {} MBS".format(func.__name__,(mem_after-mem_before)/1024**2))
        return result
    return wrapper

def clear_cache(func):
    def wrapper(*args,**kwargs):
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        result = func(*args,**kwargs)
        gc.collect()
        mem_after = process.memory_info().rss
        print("Memory cleared is {} MBS".format((mem_after-mem_before)/1024**2))
        return result
    return wrapper


            


