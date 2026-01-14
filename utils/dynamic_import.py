import importlib


# Cache for dynamic imports to avoid repeated imports
_import_cache = {}

# === Jaccard Elbow Method ===
# Dynamic imports for custom clustering algorithms
def dynamic_import_jaccard_elbow(module_name, class_name):
    """Dynamically import clustering algorithm classes with caching"""
    cache_key = f"{module_name}.{class_name}"
    
    if cache_key in _import_cache:
        return _import_cache[cache_key]
    
    try:
        module = importlib.import_module(module_name)
        class_obj = getattr(module, class_name)
        _import_cache[cache_key] = class_obj
        return class_obj
    except ImportError as e:
        print(f"Warning: Could not import {module_name}.{class_name}: {e}")
        _import_cache[cache_key] = None
        return None
