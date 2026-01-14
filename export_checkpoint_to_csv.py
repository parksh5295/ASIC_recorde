#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export checkpoint files to CSV format
Converts pickle checkpoint files (history.pkl, all_valid_signatures.pkl, etc.) to CSV files
in the same format as the normal completion output.
"""

import os
import sys
import argparse
import pickle
import json
import pandas as pd
import numpy as np

# Add project root to path
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
except NameError:
    if '..' not in sys.path:
        sys.path.insert(0, '..')

from Dataset_Choose_Rule.isv_checkpoint_handler import (
    CHECKPOINT_FILENAME,
    SIGNATURES_FILENAME,
    BLACKLIST_FILENAME,
    HISTORY_FILENAME,
    TURNMAP_FILENAME
)

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_checkpoint_data(run_dir):
    """Load all checkpoint data from the run directory."""
    checkpoint_path = os.path.join(run_dir, CHECKPOINT_FILENAME)
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file not found: {checkpoint_path}")
        return None
    
    try:
        # Load checkpoint metadata
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        last_completed_turn = checkpoint_data.get('last_completed_turn', 0)
        
        # Load pickle files
        with open(os.path.join(run_dir, SIGNATURES_FILENAME), 'rb') as f:
            all_valid_signatures = pickle.load(f)
        
        with open(os.path.join(run_dir, BLACKLIST_FILENAME), 'rb') as f:
            signatures_to_remove = pickle.load(f)
        
        with open(os.path.join(run_dir, HISTORY_FILENAME), 'rb') as f:
            history = pickle.load(f)
        
        signature_turn_created = {}
        turnmap_path = os.path.join(run_dir, TURNMAP_FILENAME)
        if os.path.exists(turnmap_path):
            with open(turnmap_path, 'rb') as f:
                signature_turn_created = pickle.load(f)
        
        logger.info(f"Loaded checkpoint data. Last completed turn: {last_completed_turn}")
        logger.info(f"Total signatures: {len(all_valid_signatures)}, Blacklisted: {len(signatures_to_remove)}")
        logger.info(f"History entries: {len(history)}")
        
        return {
            'last_completed_turn': last_completed_turn,
            'all_valid_signatures': all_valid_signatures,
            'signatures_to_remove': signatures_to_remove,
            'history': history,
            'signature_turn_created': signature_turn_created
        }
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint data: {e}", exc_info=True)
        return None


def get_attack_columns(file_type):
    """Get attack type columns for the given file type."""
    if file_type in ['CICIDS2017']:
        return ['Label']
    if file_type in ['CICIoT2023', 'CICIoT']:
        return ['attack_name']
    if file_type in ['DARPA98', 'DARPA']:
        return ['Class']
    if file_type in ['MiraiBotnet']:
        return ['reconnaissance', 'infection', 'action']
    if file_type in ['netML']:
        return ['Label']
    if file_type in ['NSL-KDD', 'NSL_KDD']:
        return ['class']
    return []


def extract_attack_types_for_signature(sig, chunk_cache_obj, attack_cols, batch_size):
    """Extract attack types that match the signature from cached chunks."""
    attacks = set()
    if not attack_cols:
        attacks.add('-')
        return attacks
    
    try:
        # Stream through cached chunks
        for batch in chunk_cache_obj.iter_batches(batch_size=batch_size):
            if batch is None or batch.empty:
                continue
            rule_keys = [k for k in sig.keys() if k in batch.columns]
            if not rule_keys:
                continue
            match_mask = batch[rule_keys].eq(pd.Series({k: sig[k] for k in rule_keys})).all(axis=1)
            if not match_mask.any():
                continue
            matched = batch.loc[match_mask]
            
            # Mirai one-hot special handling
            if attack_cols == ['reconnaissance', 'infection', 'action']:
                for col in attack_cols:
                    if col in matched.columns:
                        vals = matched[col]
                        if (vals == 1).any():
                            attacks.add(col)
                continue
            
            for col in attack_cols:
                if col in matched.columns:
                    vals = matched[col].astype(str).unique().tolist()
                    attacks.update(vals)
    except Exception as e:
        logger.warning(f"Error extracting attack types for signature: {e}")
    
    if not attacks:
        attacks.add('-')
    return attacks


def export_history_to_csv(history, output_dir, file_type, file_number, params_str, last_completed_turn):
    """Export history to CSV in the same format as normal completion."""
    if not history:
        logger.warning("History is empty. Skipping history CSV export.")
        return
    
    try:
        history_df = pd.DataFrame(history)
        performance_filename = f"{file_type}_{file_number}_{params_str}_performance_history_eex_lastt{last_completed_turn}.csv"
        performance_path = os.path.join(output_dir, performance_filename)
        history_df.to_csv(performance_path, index=False)
        logger.info(f"Performance history saved to: {performance_path}")
        logger.info(f"  Contains {len(history_df)} turn(s)")
    except Exception as e:
        logger.error(f"Failed to export history to CSV: {e}", exc_info=True)


def export_signatures_to_csv(all_valid_signatures, signatures_to_remove, signature_turn_created,
                            output_dir, file_type, file_number, params_str, last_completed_turn,
                            chunk_cache=None, attack_cols=None, batch_size=20000):
    """Export final signatures to CSV in the same format as normal completion."""
    # Filter out blacklisted signatures
    final_signatures = {sig_id: rule for sig_id, rule in all_valid_signatures.items() 
                       if sig_id not in signatures_to_remove}
    
    if not final_signatures:
        logger.warning("No valid signatures to export.")
        return
    
    logger.info(f"Exporting {len(final_signatures)} signatures (excluding {len(signatures_to_remove)} blacklisted)")
    
    try:
        signature_records = []
        
        # If chunk_cache is provided, extract attack types
        extract_attacks = chunk_cache is not None and attack_cols is not None
        
        for sig_id, rule in final_signatures.items():
            created_turn = signature_turn_created.get(sig_id, None)
            
            if extract_attacks:
                attacks = extract_attack_types_for_signature(rule, chunk_cache, attack_cols, batch_size)
                attack_types_str = "|".join(sorted(attacks))
            else:
                attack_types_str = "-"
            
            signature_records.append({
                'signature_rule': str(rule),
                'created_turn': created_turn,
                'attack_types': attack_types_str
            })
        
        final_signatures_df = pd.DataFrame(signature_records)
        output_filename = f"{file_type}_{file_number}_{params_str}_incremental_signatures_eex_lastt{last_completed_turn}.csv"
        output_path = os.path.join(output_dir, output_filename)
        final_signatures_df.to_csv(output_path, index=False)
        logger.info(f"Final signatures saved to: {output_path}")
        logger.info(f"  Contains {len(final_signatures_df)} signature(s)")
        
    except Exception as e:
        logger.error(f"Failed to export signatures to CSV: {e}", exc_info=True)


def export_summary_metrics(history, final_signatures_count, output_dir, file_type, file_number, params_str, last_completed_turn):
    """Export summary metrics CSV."""
    try:
        if history:
            final_stats = history[-1]
            final_recall = final_stats.get('exit_recall', 0.0)
        else:
            final_recall = 0.0
        
        # Calculate average conditions per signature (we don't have the actual signatures here, so use 0)
        avg_conditions = 0.0
        
        summary_record = {
            "time_to_generate_signatures_sec": 0.0,  # Not available from checkpoint
            "num_signatures_final": final_signatures_count,
            "avg_conditions_per_signature": avg_conditions,
            "total_recall_final_turn_exit": final_recall
        }
        
        summary_df = pd.DataFrame([summary_record])
        summary_filename = f"{file_type}_{file_number}_{params_str}_summary_metrics_lastt{last_completed_turn}.csv"
        summary_path = os.path.join(output_dir, summary_filename)
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Summary metrics saved to: {summary_path}")
        
    except Exception as e:
        logger.error(f"Failed to export summary metrics: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Export checkpoint files to CSV format")
    parser.add_argument('--run_dir', type=str, required=True,
                       help="Path to the run directory containing checkpoint files")
    parser.add_argument('--file_type', type=str, required=True,
                       help="File type (e.g., MiraiBotnet, CICIoT2023)")
    parser.add_argument('--file_number', type=int, default=1,
                       help="File number (default: 1)")
    parser.add_argument('--params_str', type=str, required=True,
                       help="Parameters string (from the run directory name)")
    parser.add_argument('--with_attack_types', action='store_true',
                       help="Extract attack types from chunk cache (requires chunk_cache directory)")
    parser.add_argument('--batch_size', type=int, default=20000,
                       help="Batch size for processing chunk cache (default: 20000)")
    
    args = parser.parse_args()
    
    # Validate run directory
    if not os.path.isdir(args.run_dir):
        logger.error(f"Run directory does not exist: {args.run_dir}")
        sys.exit(1)
    
    # Load checkpoint data
    checkpoint_data = load_checkpoint_data(args.run_dir)
    if checkpoint_data is None:
        logger.error("Failed to load checkpoint data. Exiting.")
        sys.exit(1)
    
    last_completed_turn = checkpoint_data['last_completed_turn']
    
    # Export history to CSV
    export_history_to_csv(
        checkpoint_data['history'],
        args.run_dir,
        args.file_type,
        args.file_number,
        args.params_str,
        last_completed_turn
    )
    
    # Export signatures to CSV
    chunk_cache = None
    attack_cols = None
    
    if args.with_attack_types:
        try:
            from utils.chunk_cache import ChunkCache
            chunk_cache_dir = os.path.join(args.run_dir, "chunk_cache")
            if os.path.isdir(chunk_cache_dir):
                chunk_cache = ChunkCache(chunk_cache_dir)
                attack_cols = get_attack_columns(args.file_type)
                logger.info(f"Using chunk cache for attack type extraction: {chunk_cache_dir}")
            else:
                logger.warning(f"Chunk cache directory not found: {chunk_cache_dir}. Skipping attack type extraction.")
        except Exception as e:
            logger.warning(f"Failed to load chunk cache: {e}. Skipping attack type extraction.")
    
    final_signatures = {sig_id: rule for sig_id, rule in checkpoint_data['all_valid_signatures'].items() 
                       if sig_id not in checkpoint_data['signatures_to_remove']}
    
    export_signatures_to_csv(
        checkpoint_data['all_valid_signatures'],
        checkpoint_data['signatures_to_remove'],
        checkpoint_data['signature_turn_created'],
        args.run_dir,
        args.file_type,
        args.file_number,
        args.params_str,
        last_completed_turn,
        chunk_cache=chunk_cache,
        attack_cols=attack_cols,
        batch_size=args.batch_size
    )
    
    # Export summary metrics
    export_summary_metrics(
        checkpoint_data['history'],
        len(final_signatures),
        args.run_dir,
        args.file_type,
        args.file_number,
        args.params_str,
        last_completed_turn
    )
    
    logger.info("Export completed successfully!")


if __name__ == '__main__':
    main()

