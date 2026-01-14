# util for discarding preemptive columns to separate 'label' and 'label'


def select_data_without_label(data, data_type):
    if data_type == "MiraiBotnet":
        data_line = ['reconnaissance', 'infection', 'action']
    '''
    Need more setting for another data type
    '''

    # Select columns excluding 'reconnaissance', 'infection', and 'action' columns
    cols_to_change = data.columns.difference(data_line)
    # Change values ​​greater than 0 to 1
    data[cols_to_change] = (data[cols_to_change] > 0).astype(int)
    return data