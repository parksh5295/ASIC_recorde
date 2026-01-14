import logging
from multiprocessing import Process, Queue, TimeoutError
from sklearn.cluster import estimate_bandwidth

logger = logging.getLogger(__name__)

def _estimate_bandwidth_worker(X, quantile, n_samples, n_jobs, result_queue):
    """Worker function to run estimate_bandwidth in a separate process."""
    try:
        bandwidth = estimate_bandwidth(X, quantile=quantile, n_samples=n_samples, n_jobs=n_jobs)
        result_queue.put(bandwidth)
    except Exception as e:
        result_queue.put(e)

def estimate_bandwidth_with_timeout(X, initial_quantile, n_samples, procs_to_use, timeout_seconds=600):
    """
    Estimates bandwidth with a timeout and retries by increasing the quantile on failure.

    Args:
        X (np.ndarray): The data array.
        initial_quantile (float): The starting quantile for estimation.
        n_samples (int): The number of samples to use for bandwidth estimation.
        procs_to_use (int): The number of processes for n_jobs.
        timeout_seconds (int): Timeout in seconds for each attempt.

    Returns:
        float: The estimated bandwidth, or a fallback value if all attempts fail.
    """
    quantile_for_estimation = initial_quantile
    final_bandwidth = None

    while final_bandwidth is None:
        if quantile_for_estimation >= 1.0:
            logger.error(f"[DynamicBandwidth] Quantile exceeded 1.0 (final value: {quantile_for_estimation:.2f}). Aborting estimation.")
            final_bandwidth = 0.1  # Fallback
            break

        result_queue = Queue()
        worker_process = Process(
            target=_estimate_bandwidth_worker,
            args=(X, quantile_for_estimation, n_samples, procs_to_use, result_queue)
        )
        
        logger.info(f"[DynamicBandwidth] Attempting to estimate bandwidth with quantile={quantile_for_estimation:.2f} and timeout={timeout_seconds}s...")
        worker_process.start()
        
        try:
            result = result_queue.get(timeout=timeout_seconds)
            worker_process.join()

            if isinstance(result, Exception):
                raise result
                
            final_bandwidth = result
            logger.info(f"[DynamicBandwidth] Successfully estimated bandwidth: {final_bandwidth:.4f}")

        except TimeoutError:
            worker_process.terminate()
            worker_process.join()
            logger.warning(f"[DynamicBandwidth] Bandwidth estimation timed out after {timeout_seconds}s.")
            quantile_for_estimation += 0.05
            logger.warning(f"[DynamicBandwidth] Increasing quantile to {quantile_for_estimation:.2f} and retrying...")

        except Exception as e:
            logger.error(f"[DynamicBandwidth] Bandwidth estimation failed with an unexpected error: {e}")
            final_bandwidth = 0.1  # Fallback
            break

    if final_bandwidth is not None and final_bandwidth <= 0:
        logger.warning(f"[DynamicBandwidth] Estimated bandwidth is non-positive ({final_bandwidth}). Using fallback value.")
        final_bandwidth = 0.1

    return final_bandwidth
