"""
Safe argparse example (renamed to avoid clashing with standard argparse).
Use this file instead of utils/argparse.py to prevent name collisions.
"""

import argparse


def build_parser():
    parser = argparse.ArgumentParser(description='Argparser example')
    parser.add_argument('--file_type', type=str, default="MiraiBotnet")   # data file type
    parser.add_argument('--file_number', type=int, default=1)   # Detach files
    parser.add_argument('--train_test', type=int, default=0)    # train = 0, test = 1
    parser.add_argument('--heterogeneous', type=str, default="Normalized")   # Heterogeneous(Embedding) Methods
    parser.add_argument('--clustering', type=str, default="kmeans")   # Clustering Methods
    parser.add_argument('--association', type=str, default="apriori")   # Association Rule
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    print(args.file_type)
    print(args.file_number)
    print(args.train_test)
    print(args.heterogeneous)
    print(args.clustering)
    print(args.association)

