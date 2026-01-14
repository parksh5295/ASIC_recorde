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
        data['Date'] = data['Date'].astype(str).str.strip()
        data['StartTime'] = data['StartTime'].astype(str).str.strip()
        data['Duration'] = data['Duration'].astype(str).str.strip()

        regex2 = r'^\d{2}/\d{2}/\d{2}$'
        regex4 = r'^\d{2}/\d{2}/\d{4}$'
        
        mask2 = data['Date'].str.match(regex2)
        mask4 = data['Date'].str.match(regex4)

        # Helper for parallel date correction
        # def _correct_date_task(date_str_is_two_digit_tuple):
        #     date_str, is_two_digit = date_str_is_two_digit_tuple
        #     return correct_invalid_date(date_str, is_two_digit)

        if mask2.any():
            try:
                # Parallel correction for two-digit year dates
                problematic_dates_series_mask2 = data.loc[mask2, 'Date']
                tasks_mask2 = [(date_str, True) for date_str in problematic_dates_series_mask2]
                
                if tasks_mask2:
                    num_processes = multiprocessing.cpu_count()
                    with multiprocessing.Pool(processes=num_processes) as pool:
                        corrected_dates_list_mask2 = pool.map(_correct_date_task_worker, tasks_mask2, chunksize=5000)
                    data.loc[mask2, 'Date'] = pd.to_datetime(pd.Series(corrected_dates_list_mask2, index=problematic_dates_series_mask2.index), format="%m/%d/%y", errors='coerce')
                else:
                    # Fallback for safety, though tasks_mask2 should not be empty if mask2.any() is true and series is not all NaN
                    data.loc[mask2, 'Date'] = pd.to_datetime(data.loc[mask2, 'Date'], format="%m/%d/%y", errors='coerce') 

            except ValueError: # This will catch errors from pd.to_datetime or if pool processing fails unexpectedly
                print(f"[Info] ValueError during initial pd.to_datetime for mask2 or during parallel correction. Attempting sequential correction for problematic dates.")
                problematic_dates_mask2 = data.loc[mask2, 'Date']
                # Ensure we only apply to strings, as NaT/NaN might cause issues with apply(correct_invalid_date)
                string_dates_mask2 = problematic_dates_mask2[problematic_dates_mask2.apply(lambda x: isinstance(x, str))]
                corrected_strings_mask2 = string_dates_mask2.apply(lambda x: correct_invalid_date(x, True))
                data.loc[mask2, 'Date'] = pd.to_datetime(corrected_strings_mask2, format="%m/%d/%y", errors='coerce')
            except Exception as e:
                print(f"Error during parallel date (mask2) conversion: {e}. No fallback to pd.to_datetime without correction performed here.")
                # Optionally, re-raise or handle more gracefully depending on desired behavior
                # For now, we attempt a simple pd.to_datetime conversion without correction as a last resort if parallel fails badly.
                try:
                    data.loc[mask2, 'Date'] = pd.to_datetime(data.loc[mask2, 'Date'], format="%m/%d/%y", errors='coerce')
                except Exception as final_e:
                    print(f"Final attempt to convert mask2 dates also failed: {final_e}")

        if mask4.any():
            try:
                # Parallel correction for four-digit year dates
                problematic_dates_series_mask4 = data.loc[mask4, 'Date']
                tasks_mask4 = [(date_str, False) for date_str in problematic_dates_series_mask4]

                if tasks_mask4:
                    num_processes = multiprocessing.cpu_count()
                    with multiprocessing.Pool(processes=num_processes) as pool:
                        corrected_dates_list_mask4 = pool.map(_correct_date_task_worker, tasks_mask4, chunksize=5000)
                    data.loc[mask4, 'Date'] = pd.to_datetime(pd.Series(corrected_dates_list_mask4, index=problematic_dates_series_mask4.index), format="%m/%d/%Y", errors='coerce')
                else:
                    data.loc[mask4, 'Date'] = pd.to_datetime(data.loc[mask4, 'Date'], format="%m/%d/%Y", errors='coerce')

            except ValueError:
                print(f"[Info] ValueError during initial pd.to_datetime for mask4 or during parallel correction. Attempting sequential correction for problematic dates.")
                problematic_dates_mask4 = data.loc[mask4, 'Date']
                string_dates_mask4 = problematic_dates_mask4[problematic_dates_mask4.apply(lambda x: isinstance(x, str))]
                corrected_strings_mask4 = string_dates_mask4.apply(lambda x: correct_invalid_date(x, False))
                data.loc[mask4, 'Date'] = pd.to_datetime(corrected_strings_mask4, format="%m/%d/%Y", errors='coerce')
            except Exception as e:
                print(f"Error during parallel date (mask4) conversion: {e}. No fallback to pd.to_datetime without correction performed here.")
                try:
                    data.loc[mask4, 'Date'] = pd.to_datetime(data.loc[mask4, 'Date'], format="%m/%d/%Y", errors='coerce')
                except Exception as final_e:
                    print(f"Final attempt to convert mask4 dates also failed: {final_e}")

        data['Date_scalar'] = data['Date']

        # Parallel processing for time and duration conversions
        # Ensure columns exist before processing
        tasks_start_time = []
        if 'StartTime' in data.columns:
            tasks_start_time = data['StartTime'].tolist()
        
        tasks_duration = []
        if 'Duration' in data.columns:
            tasks_duration = data['Duration'].tolist()

        num_processes = multiprocessing.cpu_count()

        if tasks_start_time:
            try:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    data['StartTime_scalar'] = pool.map(hms_to_seconds, tasks_start_time, chunksize=5000)
            except Exception as e:
                print(f"Error during parallel StartTime conversion: {e}. Falling back to sequential.")
                data['StartTime_scalar'] = data['StartTime'].apply(hms_to_seconds)
        elif 'StartTime' in data.columns: # Column exists but no tasks (e.g. all NaN)
             data['StartTime_scalar'] = np.nan

        if tasks_duration:
            try:
                with multiprocessing.Pool(processes=num_processes) as pool:
                    data['Duration_scalar'] = pool.map(hms_to_seconds, tasks_duration, chunksize=5000)
            except Exception as e:
                print(f"Error during parallel Duration conversion: {e}. Falling back to sequential.")
                data['Duration_scalar'] = data['Duration'].apply(hms_to_seconds)
        elif 'Duration' in data.columns: # Column exists but no tasks
            data['Duration_scalar'] = np.nan

        return data
    
    # Handle CICModbus23: split combined Timestamp into date and time scalars
    elif file_type in ['CICModbus23', 'CICModbus']:
        # Parse the combined Timestamp field
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