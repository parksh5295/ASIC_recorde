import pandas as pd
import multiprocessing
import logging
from tqdm import tqdm
import signal
import os
import sys
from array import array


logger = logging.getLogger(__name__)

# This will be the global data for each worker to avoid serialization overhead
_worker_data_perf = ()

def _suppress_broken_pipe_tracebacks():
    def _handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, BrokenPipeError):
            return
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
    sys.excepthook = _handler

def _init_worker_perf(data_df):
    """Initializes each worker process with the global dataframe."""
    # Suppress BrokenPipeError messages if the main process dies.
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    _suppress_broken_pipe_tracebacks()
    global _worker_data_perf
    _worker_data_perf = data_df

def _apply_single_signature(signature):
    """
    Worker function to apply a single signature to the global data and find all matching indices.
    """
    global _worker_data_perf
    data_df = _worker_data_perf
    
    sig_id = signature['id']
    rule = signature['rule_dict']
    
    if not rule or data_df.empty:
        return sig_id, set()

    # Build a boolean mask for the rule
    mask = pd.Series([True] * len(data_df), index=data_df.index)
    try:
        for key, value in rule.items():
            if key in data_df.columns:
                # --- NEW: Handle OR conditions represented by sets/frozensets ---
                if isinstance(value, (set, frozenset)):
                    mask &= data_df[key].isin(value)
                else:
                # --- END NEW ---
                    mask &= (data_df[key] == value)
            else:
                # If a feature from the rule doesn't exist in the data, it can't match anything.
                return sig_id, set()
    except Exception as e:
        logger.error(f"Error applying rule for sig_id {sig_id}: {e}")
        return sig_id, set()

    alerted_indices = set(data_df.index[mask])
    return sig_id, alerted_indices

def _get_data_iterator_factory(data_source, data_batch_size):
    """
    Returns a callable that yields DataFrame batches each time it's invoked.
    """
    if isinstance(data_source, pd.DataFrame):
        def iterator():
            if not data_source.empty:
                yield data_source
        return iterator

    if hasattr(data_source, "iter_batches"):
        def iterator():
            yield from data_source.iter_batches(batch_size=data_batch_size)
        return iterator

    if callable(data_source):
        return data_source

    raise TypeError(f"Unsupported data source type for signature performance calculation: {type(data_source)}")


def calculate_signatures_performance(
    signatures: list,
    data_source,
    num_processes: int,
    data_batch_size: int = 50000
):
    """
    Calculates the performance of each signature on the provided dataset.
    The dataset can be a full DataFrame or any object that exposes an iterator
    yielding DataFrame batches (e.g., a ChunkCache).

    Args:
        signatures (list): [{'id': ..., 'rule_dict': ...}, ...]
        data_source: DataFrame or streaming provider with iter_batches().
        num_processes (int): Processes to use for parallel evaluation.
        data_batch_size (int): Target number of rows per streamed batch.

    Returns:
        dict: {sig_id: {'tp_indices': {...}, 'fp_indices': {...}}}
    """
    if not signatures or data_source is None:
        return {}

    iterator_factory = _get_data_iterator_factory(data_source, data_batch_size)
    performance_results = {}

    total_rows = 0
    chunk_counter = 0

    for batch_df in iterator_factory():
        if batch_df is None or batch_df.empty:
            continue

        chunk_counter += 1
        total_rows += len(batch_df)

        logger.info(f"[PerfCalc] Processing chunk {chunk_counter} with {len(batch_df)} rows for {len(signatures)} signatures...")

        chunk_positive_indices = set(batch_df[batch_df['label'] == 1].index)
        chunk_negative_indices = set(batch_df[batch_df['label'] == 0].index)

        pool_processes = num_processes or os.cpu_count() or 1
        chunk_size = max(1, len(signatures) // (pool_processes * 4)) if pool_processes > 0 else 1

        with multiprocessing.Pool(processes=pool_processes, initializer=_init_worker_perf, initargs=(batch_df,)) as pool:
            results_iterator = pool.imap_unordered(_apply_single_signature, signatures, chunksize=chunk_size)
            progress_bar = tqdm(results_iterator, total=len(signatures), desc=f"[PerfCalc] Chunk {chunk_counter}", leave=False)

            for sig_id, alerted_indices in progress_bar:
                if not alerted_indices:
                    continue

                tp_indices = alerted_indices.intersection(chunk_positive_indices)
                fp_indices = alerted_indices.intersection(chunk_negative_indices)

                if tp_indices or fp_indices:
                    entry = performance_results.setdefault(
                        sig_id,
                        {'tp_indices': array('I'), 'fp_indices': array('I')}
                    )
                    if tp_indices:
                        entry['tp_indices'].extend(sorted(tp_indices))
                    if fp_indices:
                        entry['fp_indices'].extend(sorted(fp_indices))

    logger.info(f"[PerfCalc] Processed {total_rows} total rows across {chunk_counter} streamed chunk(s).")
    logger.info(f"[PerfCalc] Performance calculation complete for {len(performance_results)} signatures that generated alerts.")
    return performance_results


def get_index_set(performance_entry: dict, key: str) -> set:
    """
    Converts a stored index collection (array/list/set) into a Python set for computation.
    """
    if not performance_entry:
        return set()
    data = performance_entry.get(key)
    if not data:
        return set()
    if isinstance(data, set):
        return data
    return set(data)
