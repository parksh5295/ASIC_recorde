# Convert date, time to integer

import pandas as pd
import numpy as np
import multiprocessing # Added for parallel processing

# Time string to seconds - Moved to top level for multiprocessing
def hms_to_seconds(hms_str):
    if pd.isna(hms_str): # Handle potential NaN values
        return np.nan
    try:
        parts = [int(x) for x in str(hms_str).strip().split(":")]
        while len(parts) < 3:
            parts = [0] + parts
        h, m, s = parts
        return h * 3600 + m * 60 + s
    except ValueError:
        return np.nan # Or handle error as appropriate

# Worker function for parallel date correction (moved from time_scalar_transfer)
def _correct_date_task_worker(date_str_is_two_digit_tuple):
    date_str, is_two_digit = date_str_is_two_digit_tuple
    return correct_invalid_date(date_str, is_two_digit)

def correct_invalid_date(date_str, is_two_digit_year=False):
    """Fix invalid dates to the nearest valid date"""
    try:
        month, day, year = map(int, date_str.split('/'))
        # Last date of each month
        month_last_day = {
            1: 31, 2: 29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
            3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }
        
        # Date calibration
        if day > month_last_day[month]:
            day = month_last_day[month]
            
        if is_two_digit_year:
            return f"{month:02d}/{day:02d}/{year:02d}"
        else:
            return f"{month:02d}/{day:02d}/{year:04d}"
    except:
        return date_str

def time_scalar_transfer(data, file_type):
    if file_type in ['DARPA', 'DARPA98']:
        # Ensure columns are treated as strings and stripped of whitespace
        if 'Date' in data.columns:
            data['Date'] = data['Date'].astype(str).str.strip()
        if 'StartTime' in data.columns:
            data['StartTime'] = data['StartTime'].astype(str).str.strip()
        if 'Duration' in data.columns:
            data['Duration'] = data['Duration'].astype(str).str.strip()

        # Robustly convert Date and StartTime to a combined timestamp
        # Using errors='coerce' will turn unparseable dates/times into NaT (Not a Time)
        if 'Date' in data.columns and 'StartTime' in data.columns:
            combined_datetime_str = data['Date'] + ' ' + data['StartTime']
            combined_datetime = pd.to_datetime(combined_datetime_str, format='%m/%d %H:%M:%S', errors='coerce')

            # Fill any NaT values with a default (e.g., epoch 0) before converting to timestamp
            epoch_start = pd.Timestamp("1970-01-01")
            data['Date'] = combined_datetime.fillna(epoch_start).apply(lambda x: x.timestamp())
            
            # Also handle StartTime conversion robustly, converting to total seconds from midnight
            start_time_td = pd.to_timedelta(data['StartTime'], errors='coerce')
            data['StartTime'] = start_time_td.fillna(pd.Timedelta(seconds=0)).dt.total_seconds()

        # Robustly convert Duration to seconds, coercing errors to NaN then filling with 0
        if 'Duration' in data.columns:
            data['Duration'] = pd.to_numeric(data['Duration'], errors='coerce').fillna(0)
        
        data.rename(columns={'Date': 'Date_scalar', 'StartTime': 'StartTime_scalar', 'Duration': 'Duration_scalar'}, inplace=True)
        
    # Handle CICModbus23: split combined Timestamp into date and time scalars
    elif file_type in ['CICModbus23', 'CICModbus']:
        if 'Timestamp' in data.columns:
            # Convert to datetime objects, coercing errors
            data['Timestamp'] = pd.to_datetime(
                data['Timestamp'], format="%Y-%m-%d %H:%M:%S.%f", errors='coerce'
            )
            # Date_scalar: full datetime for scaling
            data['Date_scalar'] = data['Timestamp']
            # StartTime_scalar: seconds since midnight
            data['StartTime_scalar'] = (
                data['Timestamp'].dt.hour * 3600
                + data['Timestamp'].dt.minute * 60
                + data['Timestamp'].dt.second
                + data['Timestamp'].dt.microsecond / 1e6
            )
            return data
        
    else:
        if 'Timestamp' in data.columns:
            data['Timestamp'] = pd.to_datetime(data['Timestamp'], errors='coerce')
            data.rename(columns={'Timestamp': 'Time_related_feature_1'}, inplace=True)
            if 'Time_related_feature_1' in data.columns:
                data['Time_related_feature_1'] = data['Time_related_feature_1'].apply(lambda x: x.timestamp() if pd.notna(x) else 0)
        else:
            pass

    return data


def convert_cic_time_to_numeric_scalars(data_df):
    """
    Converts 'Timestamp' (assumed to be datetime objects) in the DataFrame
    to numeric 'Date_scalar' (seconds from the first timestamp in the series)
    and 'StartTime_scalar' (seconds from midnight).
    This is specifically for preparing CICModbus23-like data for numeric mapping.
    It overwrites 'Date_scalar' and 'StartTime_scalar' if they exist, or creates them.
    The input DataFrame is modified.
    """
    # Operate directly on data_df, assuming caller passes a copy if original should be preserved.
    # df = data_df.copy() # Optional: if the function should always work on a copy.

    if 'Timestamp' not in data_df.columns:
        print("Error in convert_cic_time_to_numeric_scalars: 'Timestamp' column missing.")
        return data_df 

    # Ensure Timestamp is datetime, as this function relies on dt accessor.
    # The original time_scalar_transfer for CICModbus should have converted 'Timestamp' to datetime.
    # This is a safeguard. If 'Timestamp' is not datetime, conversion might fail or lead to errors.
    if not pd.api.types.is_datetime64_any_dtype(data_df['Timestamp']):
        data_df['Timestamp'] = pd.to_datetime(data_df['Timestamp'], errors='coerce')
    
    # Date_scalar: seconds from the first valid timestamp in the current data
    if data_df['Timestamp'].notna().any(): # Check if there's any non-NaT timestamp
        min_timestamp = data_df['Timestamp'].dropna().min() # Get min from non-NaT values
        data_df['Date_scalar'] = (data_df['Timestamp'] - min_timestamp).dt.total_seconds()
    else:
        data_df['Date_scalar'] = np.nan 
            
    # StartTime_scalar: seconds since midnight
    data_df['StartTime_scalar'] = (
        data_df['Timestamp'].dt.hour * 3600
        + data_df['Timestamp'].dt.minute * 60
        + data_df['Timestamp'].dt.second
        + data_df['Timestamp'].dt.microsecond / 1e6
    )
    
    # Ensure scalars are NaN if Timestamp was NaT (or became NaT after coerce)
    nan_timestamp_mask = data_df['Timestamp'].isna()
    data_df.loc[nan_timestamp_mask, 'Date_scalar'] = np.nan
    data_df.loc[nan_timestamp_mask, 'StartTime_scalar'] = np.nan
    
    return data_df