# Input embedded_data (after separate_bin)

import pandas as pd
from Heterogeneous_Method.build_interval_mapping import build_interval_mapping_dataframe
import multiprocessing # Added for parallel processing
import re # --- NEW: Import re for parsing interval strings ---

# Helper function for parallel interval mapping
def _map_single_interval_column(args):
    col_name, col_data, col_mapping_series = args
    try:
        # Check if data contains interval strings (like "(5.0, 8.0]")
        print(f"[DEBUG] col_name: {col_name}, dtype: {col_data.dtype}")
        print(f"[DEBUG] col_data sample: {col_data.head()}")
        
        # Convert category dtype to object for string operations
        if col_data.dtype.name == 'category':
            col_data = col_data.astype(str)
            print(f"[DEBUG] Converted category to string, new dtype: {col_data.dtype}")
        
        # Define regex pattern outside f-string to avoid backslash issue
        interval_pattern = r'\(.*\]'
        contains_intervals = col_data.str.contains(interval_pattern, na=False).any()
        print(f"[DEBUG] Contains interval strings: {contains_intervals}")
        
        if col_data.dtype == 'object' and contains_intervals:
            # For interval strings, we need to map them directly to group numbers
            # Create a mapping from interval strings to group numbers
            interval_to_group = {}
            for s in col_mapping_series.dropna():
                try:
                    interval_str, group_num_str = s.split('=')
                    group_num = int(group_num_str.strip())
                    interval_to_group[interval_str.strip()] = group_num
                except:
                    continue
            
            # Map interval strings directly to group numbers
            mapped_series = col_data.map(interval_to_group)
            final_mapping_info = {str(k): v for k, v in interval_to_group.items()}
            print(f"[SUCCESS] Successfully mapped {col_name} with {len(final_mapping_info)} intervals")
            return col_name, mapped_series, final_mapping_info
        else:
            # Ensure input data is numeric, coercing errors to NaN, then filling with 0.
            numeric_col_data = pd.to_numeric(col_data, errors='coerce').fillna(0)

        # Step 1: Parse the string mapping back into a list of Interval objects and group numbers
        parsed_mapping = []
        for s in col_mapping_series.dropna():
            try:
                interval_str, group_num_str = s.split('=')
                group_num = int(group_num_str.strip())
                
                interval_str = interval_str.strip()
                left_bracket = interval_str[0]
                right_bracket = interval_str[-1]
                
                #nums = [float(f) for f in re.findall(r'-?\\d+\\.?\\d*e?[-+]?\\d*', interval_str)]
                # Try multiple regex patterns to extract numbers
                nums = []
                # Pattern 1: Standard decimal numbers
                pattern1 = r'-?\d+\.?\d*'
                matches1 = re.findall(pattern1, interval_str)
                matches2 = []  # Initialize matches2 to avoid UnboundLocalError
                
                if len(matches1) >= 2:
                    nums = [float(f) for f in matches1[:2]]
                else:
                    # Pattern 2: More flexible pattern
                    pattern2 = r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
                    matches2 = re.findall(pattern2, interval_str)
                    if len(matches2) >= 2:
                        nums = [float(f) for f in matches2[:2]]
                
                if len(nums) != 2: 
                    continue
                
                closed = 'neither'
                if left_bracket == '[' and right_bracket == ']': closed = 'both'
                elif left_bracket == '[': closed = 'left'
                elif right_bracket == ']': closed = 'right'
                
                interval = pd.Interval(nums[0], nums[1], closed=closed)
                parsed_mapping.append({'interval': interval, 'group': group_num})
            except Exception as e:
                continue

        if not parsed_mapping:
            print(f"[WARNING] No parsed_mapping for {col_name}, returning NaN series")
            return col_name, pd.Series(float('nan'), index=numeric_col_data.index), {}

        # Step 2: Create a pd.IntervalIndex for efficient mapping
        parsed_mapping.sort(key=lambda x: x['interval'].left)
        intervals = [d['interval'] for d in parsed_mapping]
        interval_index = pd.IntervalIndex(intervals)

        # Step 3: Use pd.cut to find which interval each data point belongs to
        # This returns a Series containing pd.Interval objects for each row
        binned_series_with_intervals = pd.cut(numeric_col_data, bins=interval_index)

        # Step 4: Map the resulting pd.Interval objects to their corresponding group numbers
        interval_to_group_map = {d['interval']: d['group'] for d in parsed_mapping}
        mapped_series = binned_series_with_intervals.map(interval_to_group_map)

        final_mapping_info = {str(k): v for k, v in interval_to_group_map.items()}
        print(f"[SUCCESS] Successfully mapped {col_name} with {len(final_mapping_info)} intervals")
        return col_name, mapped_series, final_mapping_info
        
    except Exception as e:
        # Fallback for any unexpected error during the process
        return col_name, pd.Series(float('nan'), index=col_data.index), {}


def map_intervals_to_groups(df, category_mapping, data_list, regul='N'):
    # mapped_df = pd.DataFrame() # Initialize later from results
    mapping_info = {}   # Save per-feature mapping information
    
    print(f"[DEBUG] Input df NaN count before map_intervals_to_groups: {df.isnull().sum().to_dict()}")

    # Ensure 'interval' key exists and is a DataFrame
    if 'interval' not in category_mapping or not isinstance(category_mapping['interval'], pd.DataFrame):
        # If no interval mapping, or mapping is not in expected format, handle appropriately
        # For now, assume interval_columns would be empty, or raise error
        print("Warning/Error: 'interval' mapping is missing or not a DataFrame in category_mapping.")
        interval_columns = []
        interval_df = pd.DataFrame(index=df.index) # Empty DataFrame with original index
    else:
        interval_columns = [col for col in category_mapping['interval'].columns if col in df.columns]
        if not interval_columns:
            print("No common columns found between df and interval_mapping for interval processing.")
            interval_df = pd.DataFrame(index=df.index) # Empty DataFrame with original index
        else:
            interval_df = df[interval_columns]  # Organize only the conditions that want to map

    interval_mapping_df = category_mapping.get('interval', pd.DataFrame()) # interval information is taken from here

    tasks = []
    for col in interval_df.columns: # Iterate over columns present in both df and interval_mapping_df
        if col not in interval_mapping_df.columns:
            # This case should be less likely now due to pre-filtering interval_columns
            # print(f"Warning: Interval mapping for column `{col}` is missing in interval_mapping_df. Skipping.")
            # mapped_df[col] = interval_df[col] # Keep original if no mapping? Or NaN?
            continue 
        tasks.append((col, interval_df[col], interval_mapping_df[col]))

    processed_columns = {}
    if tasks:
        num_processes = min(len(tasks), multiprocessing.cpu_count())
        # print(f"[MapIntervals] Using {num_processes} processes for {len(tasks)} interval columns.")
        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                results = pool.map(_map_single_interval_column, tasks)
        except Exception as e:
            print(f"Error during parallel interval mapping: {e}. Falling back to sequential.")
            results = [_map_single_interval_column(task) for task in tasks]
        
        for col_name, mapped_series, i_to_g in results:
            processed_columns[col_name] = mapped_series
            mapping_info[col_name] = i_to_g
    
    if processed_columns:
        mapped_df_from_intervals = pd.DataFrame(processed_columns, index=interval_df.index) # Ensure correct index
    else:
        # Ensure mapped_df is initialized correctly if no interval columns were processed
        mapped_df_from_intervals = pd.DataFrame(index=df.index) 

    # Concatenation logic (ensure data_list[0] and data_list[-1] have compatible indices)
    # It is assumed that data_list[0] (categorical) and data_list[-1] (binary) are already processed 
    # and have an index compatible with df and thus with mapped_df_from_intervals.
    # If data_list elements are empty DataFrames, they should also have the correct index.
    
    # Ensure all parts have an index, default to df.index if they are empty and don't have one
    part1 = data_list[0] if not data_list[0].empty else pd.DataFrame(index=df.index)
    part3 = data_list[len(data_list)-1] if not data_list[len(data_list)-1].empty else pd.DataFrame(index=df.index)

    # Ensure mapped_df_from_intervals also has the correct index if it's empty
    if mapped_df_from_intervals.empty and not processed_columns:
        mapped_df_from_intervals = pd.DataFrame(index=df.index)

    # Filter out empty DataFrames before concat to avoid issues if one part is truly empty of columns but has an index.
    # Concat will handle empty DFs gracefully if they have an index. The main concern is if they are None or lack an index.
    dfs_to_concat = []
    if not part1.empty or part1.shape[1] > 0: dfs_to_concat.append(part1)
    if not mapped_df_from_intervals.empty or mapped_df_from_intervals.shape[1] > 0: dfs_to_concat.append(mapped_df_from_intervals)
    if not part3.empty or part3.shape[1] > 0: dfs_to_concat.append(part3)
    
    if dfs_to_concat:
        mapped_df = pd.concat(dfs_to_concat, axis=1)
    else: # All parts were empty
        mapped_df = pd.DataFrame(index=df.index)

    print(f"[DEBUG] Output mapped_df NaN count after map_intervals_to_groups: {mapped_df.isnull().sum().to_dict()}")
    
    mapped_info_df = build_interval_mapping_dataframe(mapping_info)

    # Update category_mapping['interval'] with the new mapping info
    if 'interval' in category_mapping and not mapped_info_df.empty:
        # Update the interval mapping with the new mapping info
        category_mapping['interval'] = mapped_info_df
        print(f"[SUCCESS] Updated category_mapping['interval'] with new mapping info")
        
        # Debug: Check for NaN values in updated interval mapping
        nan_count = category_mapping['interval'].isnull().sum().sum()
        total_elements = category_mapping['interval'].size
        print(f"[DEBUG] Updated interval mapping NaN count: {nan_count} out of {total_elements} elements")
        if nan_count > 0:
            print(f"[DEBUG] NaN columns in updated interval mapping: {category_mapping['interval'].isnull().any().to_dict()}")
        else:
            print(f"[DEBUG] No NaN values in updated interval mapping!")

    if regul in ["N", "n"]:
        mapped_info_df = pd.concat([
            category_mapping.get('categorical', pd.DataFrame()),
            mapped_info_df,
            category_mapping.get('binary', pd.DataFrame())
        ], axis=1, ignore_index=False)

    return mapped_df, mapped_info_df
