"""Bound MPS resources in the serial live-search worker."""
import gc
import time


def run_live_search(engine, function, *args, **kwargs):
    if getattr(getattr(engine, 'device', None), 'type', None) != 'mps':
        return function(*args, **kwargs)
    output, error = None, None
    matcher = getattr(engine, 'matcher', None)
    attention = [(module, module.has_sdp) for module in matcher.modules()
                 if hasattr(module, 'has_sdp')] if matcher is not None else []
    # LightGlue already supplies this mathematically equivalent attention path.
    # MPS SDPA grew driver memory across variable-sized queries in our stress run.
    for module, _ in attention:
        module.has_sdp = False
    try:
        output = function(*args, **kwargs)
    except Exception as exc:
        # Do not put an inference traceback (and its dense tensors) into Future.
        # Raising a fresh error below, OUTSIDE this except, releases that graph.
        error = f'{type(exc).__name__}: {exc}'
    finally:
        for module, previous in attention:
            module.has_sdp = previous
    started = time.monotonic()
    memory = dict(failed=error is not None, attention_backend='explicit')
    try:
        if error is not None:
            gc.collect()
        mps = engine.torch.mps
        mps.synchronize()
        memory.update(current_before_bytes=mps.current_allocated_memory(),
                      driver_before_bytes=mps.driver_allocated_memory())
        mps.empty_cache()
        mps.synchronize()
        memory.update(current_after_bytes=mps.current_allocated_memory(),
                      driver_after_bytes=mps.driver_allocated_memory())
    except Exception as exc:
        memory['cleanup_error'] = str(exc)
        if error is None:
            error = f'MPS resource cleanup failed: {exc}'
    memory['cleanup_ms'] = (time.monotonic()-started)*1000
    sink = getattr(engine, 'diagnostic_sink', None)
    if sink is not None:
        sink('search_resources', **getattr(engine, 'diagnostic_context', {}), **memory)
    if error is not None:
        raise RuntimeError(error)
    return output
