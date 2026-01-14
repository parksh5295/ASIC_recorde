import os
import pickle
import json
import logging

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = "_checkpoint.json"
SIGNATURES_FILENAME = "all_valid_signatures.pkl"
BLACKLIST_FILENAME = "signatures_to_remove.pkl"
HISTORY_FILENAME = "history.pkl"
TURNMAP_FILENAME = "signature_turn_created.pkl"

def save_checkpoint(run_dir, turn_counter, all_valid_signatures, signatures_to_remove, history, signature_turn_created=None):
    """Saves the state of the incremental validation process."""
    try:
        os.makedirs(run_dir, exist_ok=True)

        # Save main data structures using pickle for efficiency
        with open(os.path.join(run_dir, SIGNATURES_FILENAME), 'wb') as f:
            pickle.dump(all_valid_signatures, f)
        
        with open(os.path.join(run_dir, BLACKLIST_FILENAME), 'wb') as f:
            pickle.dump(signatures_to_remove, f)
            
        with open(os.path.join(run_dir, HISTORY_FILENAME), 'wb') as f:
            pickle.dump(history, f)

        if signature_turn_created is not None:
            with open(os.path.join(run_dir, TURNMAP_FILENAME), 'wb') as f:
                pickle.dump(signature_turn_created, f)

        # Save the last completed turn number
        with open(os.path.join(run_dir, CHECKPOINT_FILENAME), 'w') as f:
            json.dump({'last_completed_turn': turn_counter}, f)
        
        logger.info(f"Checkpoint saved for turn {turn_counter} in '{run_dir}'")

    except Exception as e:
        logger.error(f"Failed to save checkpoint for turn {turn_counter}: {e}", exc_info=True)


def load_checkpoint(run_dir):
    """Loads the state of the incremental validation process from a checkpoint."""
    checkpoint_path = os.path.join(run_dir, CHECKPOINT_FILENAME)
    
    if not os.path.exists(checkpoint_path):
        logger.info("No checkpoint found. Starting a new run.")
        return None

    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        last_completed_turn = checkpoint_data.get('last_completed_turn', 0)

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
            
        logger.info(f"Checkpoint loaded. Resuming from after turn {last_completed_turn}.")
        
        return {
            'resume_from_turn': last_completed_turn,
            'all_valid_signatures': all_valid_signatures,
            'signatures_to_remove': signatures_to_remove,
            'history': history,
            'signature_turn_created': signature_turn_created
        }

    except Exception as e:
        logger.error(f"Failed to load checkpoint from '{run_dir}'. Starting a new run. Error: {e}", exc_info=True)
        return None
