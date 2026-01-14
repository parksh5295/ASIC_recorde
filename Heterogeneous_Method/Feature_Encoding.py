# Separating features into attributes and encoding each
# Output data is 'feature_list'
# Encoding and Normalization

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import numpy as np
import pandas as pd
import multiprocessing # Added for parallel processing


# Helper function for parallel LabelEncoding of categorical features
def _encode_categorical_feature(args):
    feature_name, series_data = args
    original_index = series_data.index # Capture original index
    try:
        le = LabelEncoder()
        # Ensure string type for LE and preserve index
        encoded_values = le.fit_transform(series_data.astype(str))
        encoded_series = pd.Series(encoded_values, name=feature_name, index=original_index)
        return feature_name, encoded_series, le
    except Exception as e:
        print(f"Error encoding categorical feature {feature_name}: {e}")
        # Return original series (which already has index) and None for encoder
        return feature_name, series_data, None

# Helper function for parallel scaling of numerical feature groups
def _scale_numerical_feature_group(args):
    data_subset_df, group_feature_list, group_name = args
    try:
        if not group_feature_list or data_subset_df.empty: # Check if there are features to scale
            return group_name, pd.DataFrame(index=data_subset_df.index) # Return empty DF with original index
        
        # Ensure all columns in group_feature_list exist in data_subset_df
        missing_cols = [col for col in group_feature_list if col not in data_subset_df.columns]
        if missing_cols:
            # print(f"Warning: Columns {missing_cols} not found in data_subset_df for group {group_name}. Skipping them.")
            group_feature_list = [col for col in group_feature_list if col in data_subset_df.columns]
            if not group_feature_list: # If all columns were missing
                 return group_name, pd.DataFrame(index=data_subset_df.index)

        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_subset_df[group_feature_list])
        scaled_df = pd.DataFrame(scaled_data, columns=group_feature_list, index=data_subset_df.index)
        return group_name, scaled_df
    except Exception as e:
        print(f"Error scaling numerical feature group {group_name}: {e}")
        # Return original subset to avoid breaking concatenation, but it won't be scaled.
        return group_name, data_subset_df[group_feature_list] if group_feature_list and not data_subset_df.empty else pd.DataFrame(index=data_subset_df.index)


def Heterogeneous_Feature_named_featrues(file_type):
    # Initialize feature lists to empty lists to prevent UnboundLocalError
    categorical_features = []
    time_features = []
    packet_length_features = []
    count_features = []
    binary_features = []

    if file_type == 'MiraiBotnet':
        categorical_features = [
            'flow_protocol'
            ]
        time_features = [
            'flow_iat_max', 'flow_iat_min', 'flow_iat_mean', 'flow_iat_total', 'flow_iat_std',
            'forward_iat_max', 'forward_iat_min', 'forward_iat_mean', 'forward_iat_total', 'forward_iat_std',
            'backward_iat_max', 'backward_iat_min', 'backward_iat_mean', 'backward_iat_total', 'backward_iat_std'
            ]
        packet_length_features = [
            'total_bhlen', 'total_fhlen',
            'forward_packet_length_mean', 'forward_packet_length_min', 'forward_packet_length_max', 'forward_packet_length_std',
            'backward_packet_length_mean', 'backward_packet_length_min', 'backward_packet_length_max', 'backward_packet_length_std'
        ]
        count_features = [
            'fpkts_per_second', 'bpkts_per_second', 'total_forward_packets', 'total_backward_packets',
            'total_length_of_forward_packets', 'total_length_of_backward_packets', 'flow_packets_per_second'
        ]
        binary_features = [
            'flow_psh', 'flow_syn', 'flow_urg', 'flow_fin', 'flow_ece', 'flow_ack', 'flow_rst', 'flow_cwr'
        ]

    elif file_type in ['MitM', 'Kitsune']:
        categorical_features = []
        time_features = [
            # 100ms window
            'SrcMAC_IP_w_100ms', 'SrcMAC_IP_mu_100ms', 'SrcMAC_IP_sigma_100ms', 'SrcMAC_IP_max_100ms', 'SrcMAC_IP_min_100ms',
            'SrcIP_w_100ms', 'SrcIP_mu_100ms', 'SrcIP_sigma_100ms', 'SrcIP_max_100ms', 'SrcIP_min_100ms',
            'Channel_w_100ms', 'Channel_mu_100ms', 'Channel_sigma_100ms', 'Channel_max_100ms', 'Channel_min_100ms',
            # 500ms window
            'SrcMAC_IP_w_500ms', 'SrcMAC_IP_mu_500ms', 'SrcMAC_IP_sigma_500ms', 'SrcMAC_IP_max_500ms', 'SrcMAC_IP_min_500ms',
            'SrcIP_w_500ms', 'SrcIP_mu_500ms', 'SrcIP_sigma_500ms', 'SrcIP_max_500ms', 'SrcIP_min_500ms',
            'Channel_w_500ms', 'Channel_mu_500ms', 'Channel_sigma_500ms', 'Channel_max_500ms', 'Channel_min_500ms',
            # 1.5s window
            'SrcMAC_IP_w_1.5s', 'SrcMAC_IP_mu_1.5s', 'SrcMAC_IP_sigma_1.5s', 'SrcMAC_IP_max_1.5s', 'SrcMAC_IP_min_1.5s',
            'SrcIP_w_1.5s', 'SrcIP_mu_1.5s', 'SrcIP_sigma_1.5s', 'SrcIP_max_1.5s', 'SrcIP_min_1.5s',
            'Channel_w_1.5s', 'Channel_mu_1.5s', 'Channel_sigma_1.5s', 'Channel_max_1.5s', 'Channel_min_1.5s',
            # 10s window
            'SrcMAC_IP_w_10s', 'SrcMAC_IP_mu_10s', 'SrcMAC_IP_sigma_10s', 'SrcMAC_IP_max_10s', 'SrcMAC_IP_min_10s',
            'SrcIP_w_10s', 'SrcIP_mu_10s', 'SrcIP_sigma_10s', 'SrcIP_max_10s', 'SrcIP_min_10s',
            'Channel_w_10s', 'Channel_mu_10s', 'Channel_sigma_10s', 'Channel_max_10s', 'Channel_min_10s',
            # 1min window
            'SrcMAC_IP_w_1min', 'SrcMAC_IP_mu_1min', 'SrcMAC_IP_sigma_1min', 'SrcMAC_IP_max_1min', 'SrcMAC_IP_min_1min',
            'SrcIP_w_1min', 'SrcIP_mu_1min', 'SrcIP_sigma_1min', 'SrcIP_max_1min', 'SrcIP_min_1min',
            'Channel_w_1min', 'Channel_mu_1min', 'Channel_sigma_1min', 'Channel_max_1min', 'Channel_min_1min'
        ]
        packet_length_features = [
            # 100ms window
            'Socket_w_100ms', 'Socket_mu_100ms', 'Socket_sigma_100ms', 'Socket_max_100ms', 'Socket_min_100ms',
            # 500ms window
            'Socket_w_500ms', 'Socket_mu_500ms', 'Socket_sigma_500ms', 'Socket_max_500ms', 'Socket_min_500ms',
            # 1.5s window
            'Socket_w_1.5s', 'Socket_mu_1.5s', 'Socket_sigma_1.5s', 'Socket_max_1.5s', 'Socket_min_1.5s',
            # 10s window
            'Socket_w_10s', 'Socket_mu_10s', 'Socket_sigma_10s', 'Socket_max_10s', 'Socket_min_10s',
            # 1min window
            'Socket_w_1min', 'Socket_mu_1min', 'Socket_sigma_1min', 'Socket_max_1min', 'Socket_min_1min'
        ]
        count_features = [
            # 100ms window
            'Jitter_mu_100ms', 'Jitter_sigma_100ms', 'Jitter_max_100ms',
            # 500ms window
            'Jitter_mu_500ms', 'Jitter_sigma_500ms', 'Jitter_max_500ms',
            # 1.5s window
            'Jitter_mu_1.5s', 'Jitter_sigma_1.5s', 'Jitter_max_1.5s',
            # 10s window
            'Jitter_mu_10s', 'Jitter_sigma_10s', 'Jitter_max_10s',
            # 1min window
            'Jitter_mu_1min', 'Jitter_sigma_1min', 'Jitter_max_1min'
        ]
        binary_features = []

    elif file_type in ['IoTID20', 'IoTID']:
        categorical_features = [
            'Src_IP', 'Dst_IP', 'Src_Port', 'Dst_Port', 'Protocol'
        ]
        time_features = [
            'Timestamp', # Note: This feature likely requires preprocessing to be numerical (e.g., Unix timestamp or scalar components like hour, day) for scaling.
            'Flow_Duration',
            'Flow_IAT_Mean', 'Flow_IAT_Std', 'Flow_IAT_Max', 'Flow_IAT_Min',
            'Fwd_IAT_Tot', 'Fwd_IAT_Mean', 'Fwd_IAT_Std', 'Fwd_IAT_Max', 'Fwd_IAT_Min',
            'Bwd_IAT_Tot', 'Bwd_IAT_Mean', 'Bwd_IAT_Std', 'Bwd_IAT_Max', 'Bwd_IAT_Min',
            'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min',
            'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min'
        ]
        packet_length_features = [
            'TotLen_Fwd_Pkts', 'TotLen_Bwd_Pkts',
            'Fwd_Pkt_Len_Max', 'Fwd_Pkt_Len_Min', 'Fwd_Pkt_Len_Mean', 'Fwd_Pkt_Len_Std',
            'Bwd_Pkt_Len_Max', 'Bwd_Pkt_Len_Min', 'Bwd_Pkt_Len_Mean', 'Bwd_Pkt_Len_Std',
            'Pkt_Len_Min', 'Pkt_Len_Max', 'Pkt_Len_Mean', 'Pkt_Len_Std', 'Pkt_Len_Var',
            'Pkt_Size_Avg',
            'Fwd_Seg_Size_Avg', 'Bwd_Seg_Size_Avg',
            'Subflow_Fwd_Byts', 'Subflow_Bwd_Byts',
            'Init_Fwd_Win_Byts', 'Init_Bwd_Win_Byts',
            'Fwd_Seg_Size_Min'
        ]
        count_features = [
            'Tot_Fwd_Pkts', 'Tot_Bwd_Pkts',
            'Flow_Byts/s', 'Flow_Pkts/s',
            'Fwd_Header_Len', 'Bwd_Header_Len', # Consistent with CICIDS2017 categorization in this file
            'Fwd_Pkts/s', 'Bwd_Pkts/s',
            'Down/Up_Ratio',
            'Fwd_Byts/b_Avg', 'Fwd_Pkts/b_Avg', 'Fwd_Blk_Rate_Avg',
            'Bwd_Byts/b_Avg', 'Bwd_Pkts/b_Avg', 'Bwd_Blk_Rate_Avg',
            'Subflow_Fwd_Pkts', 'Subflow_Bwd_Pkts',
            'Fwd_Act_Data_Pkts'
        ]
        binary_features = [
            'Fwd_PSH_Flags', 'Bwd_PSH_Flags',
            'Fwd_URG_Flags', 'Bwd_URG_Flags',
            'FIN_Flag_Cnt', 'SYN_Flag_Cnt', 'RST_Flag_Cnt', 'PSH_Flag_Cnt',
            'ACK_Flag_Cnt', 'URG_Flag_Cnt', 'CWE_Flag_Count', 'ECE_Flag_Cnt'
        ]

    elif file_type in ['CICIDS2017', 'CICIDS']:
        categorical_features = ['Destination Port']
        time_features = [
            'Flow Duration', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ]
        packet_length_features = [
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
            'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
            'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
        ]
        count_features = [
            'Total Fwd Packets', 'Total Backward Packets',
            'Flow Bytes/s', 'Flow Packets/s',
            'Fwd Packets/s', 'Bwd Packets/s',
            'Fwd Header Length', 'Bwd Header Length',
            'Down/Up Ratio', 'Average Packet Size',
            'Fwd Avg Bytes/Bulk', 'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate',
            'Bwd Avg Bytes/Bulk', 'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate',
            'Subflow Fwd Packets', 'Subflow Fwd Bytes',
            'Subflow Bwd Packets', 'Subflow Bwd Bytes',
            'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
            'act_data_pkt_fwd', 'min_seg_size_forward'
        ]
        binary_features = [
            'Fwd PSH Flags', 'Bwd PSH Flags',
            'Fwd URG Flags', 'Bwd URG Flags',
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count',
            'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
            'CWE Flag Count', 'ECE Flag Count'
        ]

    elif file_type == 'netML':
        categorical_features = ['Protocol', 'Source IP', 'Destination IP', 'Source Port', 'Destination Port']
        time_features = [
            'Flow IAT Max', 'Flow IAT Min', 'Flow IAT Mean', 'Flow IAT Std',
            'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Max', 'Fwd IAT Min',
            'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Max', 'Bwd IAT Min',
            'Active Mean', 'Active Std', 'Active Max', 'Active Min',
            'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
        ]
        packet_length_features = [
            'Flow Duration', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
            'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std',
            'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
            'Avg Fwd Segment Size', 'Avg Bwd Segment Size'
        ]
        count_features = [
            'Total Fwd Packets', 'Total Backward Packets', 'Flow Packets/s', 'Flow Bytes/s',
            'Fwd Packets/s', 'Bwd Packets/s', 'Subflow Fwd Packets', 'Subflow Bwd Packets',
            'Init_Win_bytes_forward', 'Init_Win_bytes_backward',
            'act_data_pkt_fwd', 'min_seg_size_forward',
            'Fwd Header Length', 'Bwd Header Length',
            'Down/Up Ratio', 'Average Packet Size'
        ]
        binary_features = [
            'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
            'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
            'Fwd PSH Flags', 'Bwd PSH Flags',
            # 'Fwd URG Flags', 'Bwd URG Flags'
        ]

    elif file_type in ['NSL-KDD', 'NSL_KDD']:
        categorical_features = ['protocol_type', 'service', 'flag']
        time_features = []
        packet_length_features = [
            'src_bytes', 'dst_bytes'
        ]
        count_features = [
            'duration', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'num_compromised', 'root_shell', 'su_attempted',
            'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate'
        ]
        binary_features = [
            'land', 'logged_in', 'is_host_login', 'is_guest_login'
        ]

    elif file_type in ['DARPA', 'DARPA98']:
        categorical_features = [
            'Protocol', 'SrcPort', 'DstPort', 'SrcIP', 'DstIP'
        ]
        time_features = [
            'Date_scalar', 'StartTime_scalar', 'Duration_scalar'
        ]
        packet_length_features = [
            
        ]
        count_features = [
            
        ]
        binary_features = [
            # 'Flag'
        ]

    elif file_type in ['CICModbus23', 'CICModbus']:
        categorical_features = [
            'TargetIP', 'TransactionID'
        ]
        time_features = [
            'Date_scalar', 'StartTime_scalar'
        ]
        packet_length_features = [
            
        ]
        count_features = [
            
        ]
        binary_features = [
        
        ]
    
    elif file_type in ['CICIoT', 'CICIoT2023']:
        categorical_features = [
            'protocol type'  # Number as an ID
        ]
        time_features = [
            'flow_duration', 'duration', 'iat' # Continuous time values
        ]
        packet_length_features = [
            'header_length', 'tot sum', 'min', 'max', 'avg', 'std', 'tot size',
            'magnitue', 'radius', 'covariance', 'variance', 'weight' # Other continuous numerical values
        ]
        count_features = [
            'rate', 'srate', 'drate', 'ack_count', 'syn_count', 'fin_count',
            'urg_count', 'rst_count', 'number' # Continuous count values
        ]
        binary_features = [
            'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
            'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
            'http', 'https', 'dns', 'telnet', 'smtp', 'ssh', 'irc',
            'tcp', 'udp', 'dhcp', 'arp', 'icmp', 'ipv', 'llc' # 0/1 flags
        ]

    return {
            'categorical_features': categorical_features,
            'time_features': time_features,
            'packet_length_features': packet_length_features,
            'count_features': count_features,
            'binary_features': binary_features
        }   


def Heterogeneous_Feature_named_combine(categorical_features, time_features, packet_length_features, count_features, binary_features, data):
    encoder = LabelEncoder()
    categorical_mapping_df = pd.DataFrame()
    binary_mapping_df = pd.DataFrame()

    # If index is not unique, reset it to avoid issues during potential joins/merges later
    original_index = data.index
    df_index_to_use = data.index # Store the index to use for empty DataFrames
    # It might be safer to always work with a consistent index if concat is involved
    # data = data.reset_index(drop=True)
    # df_index_to_use = data.index


    if not categorical_features:
        categorical_data = pd.DataFrame(index=df_index_to_use)
        categorical_mapping_info = {}
    else:
        categorical_data = pd.DataFrame(index=df_index_to_use)
        categorical_mapping_info = {}
        tasks = []
        for col in categorical_features:
            if col in data.columns:
                tasks.append((col, data[col]))
            else:
                print(f"Warning: Categorical column '{col}' not found in data. Skipping.")
                categorical_data[col] = pd.NA # Assign NA to the column

        if tasks:
            num_processes_cat = min(len(tasks), multiprocessing.cpu_count())
            # print(f"[HFN_combine Cat] Using {num_processes_cat} processes for {len(tasks)} categorical features.")
            try:
                with multiprocessing.Pool(processes=num_processes_cat) as pool:
                    results = pool.map(_encode_categorical_feature, tasks)
            except Exception as e:
                print(f"Error during parallel LabelEncoding in Heterogeneous_Feature_named_combine: {e}. Falling back to sequential.")
                results = [_encode_categorical_feature(task) for task in tasks]
            
            for feature_name, encoded_series, le in results:
                if le is not None: # Successfully encoded
                    # Use .loc to assign with correct index alignment
                    categorical_data.loc[df_index_to_use, feature_name] = encoded_series + 1
                    categorical_mapping_info[feature_name] = dict(zip(le.classes_, le.transform(le.classes_) + 1))
                else: # Encoding failed, original series (or series with NA) was returned
                    if feature_name in data.columns: # if original series was returned
                         categorical_data.loc[df_index_to_use, feature_name] = encoded_series # Keep as is or NA if it was set before
                    else: # if column was skipped initially and we created an NA column
                         categorical_data[feature_name] = pd.NA

        max_len = max(len(mapping) for mapping in categorical_mapping_info.values()) if categorical_mapping_info else 0
        formatted_columns = {}
        for feature, mapping in categorical_mapping_info.items():
            items = [f"{k}={v}" for k, v in mapping.items()]
            items += [""] * (max_len - len(items))
            formatted_columns[feature] = items

        if formatted_columns:
            categorical_mapping_df = pd.DataFrame(formatted_columns)

    if not time_features:
        # Fix: Allocate empty DataFrame instead of np.empty (keep index)
        time_data = pd.DataFrame(index=df_index_to_use)
    else:
        # Ensure selection uses the correct index
        time_data = data.loc[df_index_to_use, time_features].copy()

    if not packet_length_features:
        packet_length_data = pd.DataFrame(index=df_index_to_use)
    else:
        packet_length_data = data.loc[df_index_to_use, packet_length_features].copy()

    if not count_features:
        packet_count_data = pd.DataFrame(index=df_index_to_use)
    else:
        packet_count_data = data.loc[df_index_to_use, count_features].copy()

    if not binary_features:
        flow_flag_data = pd.DataFrame(index=df_index_to_use)
        binary_mapping_info = {} # Initialize to prevent NameError if binary_features is empty
    else: # binary_features list is not empty
        # Filter binary_features to only include columns present in the data
        existing_binary_features = [col for col in binary_features if col in data.columns]

        # Use the filtered list to create flow_flag_data, preventing KeyError
        flow_flag_data = data.loc[df_index_to_use, existing_binary_features].copy()
        
        binary_mapping_info = {} # Initialize as in original code structure
        # Populate binary_mapping_info using only the existing binary features
        for col in existing_binary_features:
             binary_mapping_info[col] = {0: 0, 1: 1}

        max_len = max(len(mapping) for mapping in binary_mapping_info.values()) if binary_mapping_info else 0
        formatted_binary = {}
        for feature, mapping in binary_mapping_info.items():
            items = [f"{k}={v}" for k, v in mapping.items()]
            items += [""] * (max_len - len(items))
            formatted_binary[feature] = items

        if formatted_binary:
            binary_mapping_df = pd.DataFrame(formatted_binary)

    # Ensure all dataframes in data_list use the same index
    # No need to set index again if we used df_index_to_use consistently
    data_list = [categorical_data, time_data, packet_length_data, packet_count_data, flow_flag_data]

    category_mapping = {
        'categorical': categorical_mapping_df,
        'binary': binary_mapping_df
    }
    # print("flag_data: ", data_list[4])

    return data_list, category_mapping


def Heterogeneous_Feature_named_combine_standard(categorical_features, time_features, packet_length_features, count_features, binary_features, data):
    # Categorical Features (Label Encoding only)
    data_categorical_encoded_list = []
    if categorical_features:
        cat_tasks_std = [(feature, data[feature]) for feature in categorical_features if feature in data.columns]
        
        processed_cat_results_std = []
        if cat_tasks_std:
            num_processes_cat_std = min(len(cat_tasks_std), multiprocessing.cpu_count())
            # print(f"[HFN_standard Cat] Using {num_processes_cat_std} processes for {len(cat_tasks_std)} categorical features.")
            try:
                with multiprocessing.Pool(processes=num_processes_cat_std) as pool:
                    processed_cat_results_std = pool.map(_label_encode_feature_standard, cat_tasks_std)
            except Exception as e:
                print(f"Error during parallel LabelEncoding (standard): {e}. Falling back to sequential.")
                processed_cat_results_std = [_label_encode_feature_standard(task) for task in cat_tasks_std]

        for feature_name, encoded_series in processed_cat_results_std:
            if not encoded_series.empty: # Only append if encoding was successful (or returned non-empty)
                data_categorical_encoded_list.append(encoded_series)
            # else: print(f"Skipping empty series for feature {feature_name} in standard combine")

        if data_categorical_encoded_list:
            data_categorical = pd.concat(data_categorical_encoded_list, axis=1)
        else:
            data_categorical = pd.DataFrame(index=data.index) # Empty DF if no cat features or all failed
    else:
        data_categorical = pd.DataFrame(index=data.index)

    # Numerical Features Scaling (Time, Packet Length, Count)
    numerical_feature_groups_std = {
        "time": time_features,
        "packet_length": packet_length_features,
        "count": count_features
    }
    
    num_tasks_std = []
    for group_name, group_feature_list in numerical_feature_groups_std.items():
        if group_feature_list:
            existing_cols_in_group = [col for col in group_feature_list if col in data.columns]
            if existing_cols_in_group:
                num_tasks_std.append((data[existing_cols_in_group].copy(), existing_cols_in_group, group_name))
            # else:
                # print(f"No columns from group '{group_name}' (standard) found in data. Skipping.")

    scaled_numerical_dfs_std = {}
    if num_tasks_std:
        num_processes_num_std = min(len(num_tasks_std), multiprocessing.cpu_count())
        # print(f"[HFN_standard Num] Using {num_processes_num_std} processes for {len(num_tasks_std)} numerical groups.")
        try:
            with multiprocessing.Pool(processes=num_processes_num_std) as pool:
                scaled_results_std = pool.map(_scale_numerical_feature_group, num_tasks_std) # Reusing the same helper
        except Exception as e:
            print(f"Error during parallel numerical scaling (standard): {e}. Falling back to sequential.")
            scaled_results_std = [_scale_numerical_feature_group(task) for task in num_tasks_std]
        
        for group_name, scaled_df in scaled_results_std:
            scaled_numerical_dfs_std[group_name] = scaled_df
            
    data_time = scaled_numerical_dfs_std.get("time", pd.DataFrame(index=data.index))
    data_packet = scaled_numerical_dfs_std.get("packet_length", pd.DataFrame(index=data.index))
    data_count = scaled_numerical_dfs_std.get("count", pd.DataFrame(index=data.index))

    # Binary Features
    if binary_features:
        data_binary = data[binary_features].astype(int)
    else:
        data_binary = pd.DataFrame(index=data.index)
    
    return [data_categorical, data_time, data_packet, data_count, data_binary]


# Helper function for parallel LabelEncoding for combine_standard
def _label_encode_feature_standard(args):
    feature_name, series_data = args
    try:
        le = LabelEncoder()
        # Ensure astype(str) for LabelEncoder if features can be mixed type, though for categorical it should be fine.
        encoded_series = pd.Series(le.fit_transform(series_data.astype(str)), name=feature_name)
        return feature_name, encoded_series
    except Exception as e:
        print(f"Error label encoding feature {feature_name} (standard): {e}")
        return feature_name, pd.Series(dtype='object', name=feature_name) # Return empty series on error