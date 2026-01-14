# --- Incremental_Signature_Validation_eex.py ---
# Chunking data from SSD back into RAM for loading


import os
import shutil
import pandas as pd


class ChunkCache:
    """
    Stores processed chunks on disk while keeping track of their global indices
    so downstream steps can stream the full dataset without holding it in memory.
    """
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        if os.path.isdir(self.cache_dir):
            try:
                shutil.rmtree(self.cache_dir)
            except OSError:
                # If cleanup fails, we still try to proceed with a fresh directory
                pass
        os.makedirs(self.cache_dir, exist_ok=True)
        self.manifest = []
        self.total_rows = 0
        self.chunk_counter = 0

    def is_empty(self) -> bool:
        return self.total_rows == 0

    def register_chunk(self, df: pd.DataFrame, turn_counter: int) -> pd.DataFrame:
        """
        Assigns a global index to the provided DataFrame, persists it to disk,
        and records metadata for future streaming.
        """
        if df is None or df.empty:
            return df

        start_idx = self.total_rows
        end_idx = start_idx + len(df)
        df.index = pd.RangeIndex(start=start_idx, stop=end_idx)

        chunk_name = f"chunk_{self.chunk_counter:06d}_turn_{turn_counter}.pkl"
        chunk_path = os.path.join(self.cache_dir, chunk_name)
        df.to_pickle(chunk_path)

        self.manifest.append({
            'path': chunk_path,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'turn': turn_counter
        })

        self.total_rows = end_idx
        self.chunk_counter += 1
        return df

    def iter_batches(self, batch_size: int | None = None):
        """
        Yields DataFrames in chronological order. If batch_size is provided,
        batches will contain at most batch_size rows (combining multiple chunks
        as needed) to balance memory usage and throughput.
        """
        if batch_size is None or batch_size <= 0:
            for meta in self.manifest:
                df = pd.read_pickle(meta['path'])
                if not df.empty:
                    yield df
            return

        buffer_df = None
        for meta in self.manifest:
            df = pd.read_pickle(meta['path'])
            if df.empty:
                continue

            if buffer_df is not None:
                df = pd.concat([buffer_df, df], copy=False)
                buffer_df = None

            start = 0
            total_len = len(df)
            while start + batch_size <= total_len:
                yield df.iloc[start:start + batch_size]
                start += batch_size

            if start < total_len:
                buffer_df = df.iloc[start:]

        if buffer_df is not None and not buffer_df.empty:
            yield buffer_df


def _yield_batches_from_source(data_source, batch_size: int | None = None):
    """
    Normalizes different data sources (full DataFrame, ChunkCache, generator factory)
    into a simple iterator of DataFrames.
    """
    if data_source is None:
        return

    if isinstance(data_source, pd.DataFrame):
        if not data_source.empty:
            yield data_source
        return

    if hasattr(data_source, 'iter_batches'):
        for batch in data_source.iter_batches(batch_size=batch_size):
            if not batch.empty:
                yield batch
        return

    if callable(data_source):
        for batch in data_source():
            if batch is not None and not batch.empty:
                yield batch
        return

    raise TypeError(f"Unsupported data source type for streaming: {type(data_source)}")


