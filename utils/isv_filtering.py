import logging
import signal
import pandas as pd

logger = logging.getLogger(__name__)

RULE_SPOOL_PRESETS = {
    'netML': {'chunk_size': 1500, 'flush_threshold': 9000},
    'CICIDS2017': {'chunk_size': 4000, 'flush_threshold': 20000},
    'CICIDS': {'chunk_size': 4000, 'flush_threshold': 20000},
    'CICIoT2023': {'chunk_size': 2500, 'flush_threshold': 12500}
}

DEFAULT_RULE_SPOOL_CHUNK_SIZE = 5000
DEFAULT_RULE_SPOOL_FLUSH_MULTIPLIER = 5


def resolve_rule_spool_settings(file_type, user_chunk_size):
    preset = RULE_SPOOL_PRESETS.get(file_type, {})
    chunk_size = user_chunk_size if (user_chunk_size and user_chunk_size > 0) else preset.get('chunk_size', DEFAULT_RULE_SPOOL_CHUNK_SIZE)
    if chunk_size <= 0:
        chunk_size = DEFAULT_RULE_SPOOL_CHUNK_SIZE
    flush_threshold = preset.get('flush_threshold', chunk_size * DEFAULT_RULE_SPOOL_FLUSH_MULTIPLIER)
    if flush_threshold <= 0:
        flush_threshold = chunk_size * DEFAULT_RULE_SPOOL_FLUSH_MULTIPLIER
    flush_threshold = max(chunk_size, flush_threshold)
    return chunk_size, flush_threshold


def calculate_and_log_support_stats(df, current_min_support, turn_counter, logger_override=None):
    log = logger_override or logger
    if df.empty:
        # Check if empty due to zero rows or zero columns
        if len(df) == 0:
            log.debug(f"  [Support Analysis] Turn: {turn_counter}, Anomalous data has 0 rows. Skipping analysis.")
        elif df.shape[1] == 0:
            log.debug(f"  [Support Analysis] Turn: {turn_counter}, Anomalous data has {len(df)} rows but 0 columns (all features masked). Skipping analysis.")
        else:
            log.debug(f"  [Support Analysis] Turn: {turn_counter}, Anomalous data is empty. Skipping analysis.")
        return

    total_rows = len(df)

    unpivoted = df.melt(var_name='feature', value_name='value')
    unpivoted['value'] = unpivoted['value'].astype(str)
    item_counts = unpivoted.groupby(['feature', 'value']).size()

    item_supports = (item_counts / total_rows).sort_values(ascending=False)

    if item_supports.empty:
        log.warning(f"  [Support Analysis] Turn: {turn_counter}, No items found in anomalous data to analyze.")
        return

    max_support = item_supports.iloc[0]
    median_support = item_supports.median()
    count_above_threshold = len(item_supports[item_supports >= current_min_support])

    log.debug(f"--- Support Analysis for Turn {turn_counter} ---")
    log.debug("Top 5 most frequent items:")
    for (feature, value), support in item_supports.head(5).items():
        log.debug(f"    - Item: {{'{feature}': {value}}}, Support: {support:.4f}")

    log.debug("Bottom 5 least frequent items:")
    for (feature, value), support in item_supports.tail(5).sort_values(ascending=True).items():
        log.debug(f"    - Item: {{'{feature}': {value}}}, Support: {support:.4f}")

    log.debug(f"Support Stats: Max={max_support:.4f}, Median={median_support:.4f}")
    log.info(f"  [Support Guidance] To generate rules for this chunk, min_support must be <= {max_support:.4f}.")
    log.info(f"  [Support Guidance] There are {count_above_threshold} unique items with support >= current min_support ({current_min_support}).")

    if current_min_support > max_support:
        log.warning(f"  [Support Alert] Current min_support ({current_min_support}) is HIGHER than the max possible support ({max_support:.4f}). NO rules will be generated for this chunk.")
    log.debug(f"--- End Support Analysis ---")


def get_support_stats_for_itemset(itemset, df):
    if df is None:
        return 0, 0
    total_rows = len(df)
    if not itemset or total_rows == 0:
        return 0, total_rows

    mask = pd.Series([True] * total_rows, index=df.index)
    for key, value in itemset.items():
        if key in df.columns:
            mask &= (df[key] == value)
            if not mask.any():
                return 0, total_rows
        else:
            return 0, total_rows

    match_count = int(mask.sum())
    return match_count, total_rows


def calculate_support_for_itemset(itemset, df, min_support):
    matches, total_rows = get_support_stats_for_itemset(itemset, df)
    if total_rows == 0:
        return False
    support_ratio = matches / total_rows
    return support_ratio >= min_support


_worker_data_filter_batch = ()
_worker_data_filter = ()


def init_worker_filter_batch(normal_data_sub_chunks, min_support, negative_filtering, negative_filter_threshold):
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    total_rows = sum(len(chunk) for chunk in normal_data_sub_chunks) if normal_data_sub_chunks else 0
    global _worker_data_filter_batch
    _worker_data_filter_batch = (normal_data_sub_chunks, min_support, negative_filtering, negative_filter_threshold, total_rows)


def is_rule_valid_for_filtering_batch(rule):
    global _worker_data_filter_batch
    normal_data_sub_chunks, min_support, negative_filtering, negative_filter_threshold, total_rows = _worker_data_filter_batch

    if total_rows == 0:
        return rule

    matched_total = 0
    threshold_count = min_support * total_rows
    for sub_chunk in normal_data_sub_chunks:
        matches, _ = get_support_stats_for_itemset(rule, sub_chunk)
        matched_total += matches
        if matched_total >= threshold_count:
            return None

    support_ratio = matched_total / total_rows
    if negative_filtering and support_ratio >= negative_filter_threshold:
        return None

    return rule


def init_worker_filter(normal_data, min_support, negative_filtering, negative_filter_threshold):
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)
    global _worker_data_filter
    _worker_data_filter = (normal_data, min_support, negative_filtering, negative_filter_threshold)


def is_rule_valid_for_filtering(rule):
    global _worker_data_filter
    normal_data, min_support, negative_filtering, negative_filter_threshold = _worker_data_filter

    matches, total_rows = get_support_stats_for_itemset(rule, normal_data)
    if total_rows == 0:
        return rule

    support_ratio = matches / total_rows
    if support_ratio >= min_support:
        return None
    if negative_filtering and support_ratio >= negative_filter_threshold:
        return None
    return rule

