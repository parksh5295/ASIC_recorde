import os
import pickle
import logging
from typing import Iterable, List


logger = logging.getLogger(__name__)


class RuleSpooler:
    """
    Streams large rule batches to disk so generation and filtering can run in smaller chunks.
    """

    def __init__(self, run_dir: str, turn_counter: int, chunk_size: int = 5000):
        self.chunk_size = max(1, chunk_size)
        self.buffer: List[dict] = []
        self.files: List[str] = []
        self.total_rules = 0
        self.spool_dir = os.path.join(run_dir, "rule_spool", f"turn_{turn_counter}")
        os.makedirs(self.spool_dir, exist_ok=True)
        logger.debug(f"[RuleSpool] Initialized at {self.spool_dir} (chunk_size={self.chunk_size}).")

    def add_rules(self, rules: Iterable[dict]):
        for rule in rules:
            self.buffer.append(rule)
            self.total_rules += 1
            if len(self.buffer) >= self.chunk_size:
                self._flush()

    def _flush(self):
        if not self.buffer:
            return
        file_path = os.path.join(self.spool_dir, f"chunk_{len(self.files):06d}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(self.buffer, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.debug(f"[RuleSpool] Flushed {len(self.buffer)} rules to {file_path}.")
        self.files.append(file_path)
        self.buffer = []

    def force_flush(self):
        """
        Public hook to force an immediate flush of the in-memory buffer to disk.
        """
        self._flush()

    def buffer_length(self) -> int:
        """
        Returns the current in-memory buffer length.
        """
        return len(self.buffer)

    def finalize(self):
        self._flush()

    def has_rules(self) -> bool:
        return self.total_rules > 0
    
    def rule_count(self) -> int:
        return self.total_rules

    def consume_chunks(self):
        """
        Yields lists of rules; each chunk file is deleted immediately after reading.
        """
        self.finalize()
        for file_path in list(self.files):
            try:
                with open(file_path, 'rb') as f:
                    rules = pickle.load(f)
            finally:
                try:
                    os.remove(file_path)
                except OSError as e:
                    logger.warning(f"[RuleSpool] Failed to remove {file_path}: {e}")
            yield rules
        self.files = []

    def cleanup(self):
        """
        Removes any remaining files/buffers. Should be safe to call multiple times.
        """
        self.buffer = []
        for file_path in list(self.files):
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except OSError:
                pass
        self.files = []
        if os.path.isdir(self.spool_dir):
            try:
                os.rmdir(self.spool_dir)
            except OSError:
                # Directory may still contain other files or be in use.
                pass


