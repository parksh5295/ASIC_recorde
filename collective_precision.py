import csv
import ast
import os
import sys


csv.field_size_limit(10**8)


def count_signatures(signature_data):
    """
    Count the number of signatures in the dataset.
    """
    return len(signature_data)


def average_conditions_per_signature(signature_data):
    """
    Calculate the average number of conditions per signature.
    """
    total_conditions = sum(len(sig['signature_name']['Signature_dict']) for sig in signature_data)
    return total_conditions / len(signature_data) if signature_data else 0


def main(csv_file_path):
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            signature_data = ast.literal_eval(row['Verified_Signatures'])
            num_signatures = count_signatures(signature_data)
            avg_conditions = average_conditions_per_signature(signature_data)
            print(f"Number of Signatures: {num_signatures}")
            print(f"Average Conditions per Signature: {avg_conditions}")


main('D:\\AutoSigGen_withData\\Dataset_Paral\\signature\\MiraiBotnet\\MiraiBotnet_RARM_1_confidence_signature_train_ea15_cut40.csv')
