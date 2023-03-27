"""Argument paser for project"""
import argparse
from core import data_downloader

from pathlib import Path


def files_args(main_parser, args):
    main_parser.add_argument("--train_images",
                             default=Path(args.data_dir, "TreeImages"),
                             type=Path,
                             help="Path to training data csv.",
                             )
    main_parser.add_argument("--train_csv",
                             default=Path(args.data_dir, "Train.csv"),
                             type=Path,
                             help="Path to training data csv.")
    main_parser.add_argument("--test_csv",
                             default=Path(args.data_dir, "Test.csv"),
                             type=str,
                             help="Path to test data csv.")
    main_parser.add_argument("--sample_submission_csv",
                             default=Path(args.data_dir, "SampleSubmission.csv"),
                             type=str,
                             help="Path to sample submission csv.")
    return main_parser


def parse_project_args():
    """Argument paser for project"""
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--data_dir",
                               default="data",
                               type=str,
                               help="Folder containing project data.")
    parent_parser.add_argument("--logs_dir",
                               default="logs",
                               type=str,
                               help="Folder containing project logs.")

    subparsers = parent_parser.add_subparsers(dest="command", required=True)
    subparsers = data_downloader.setup_args(subparsers)
    subparsers.add_parser("efficientnet")

    main_parser = argparse.ArgumentParser(parents=[parent_parser])

    main_parser = files_args(main_parser, parent_parser.parse_args())

    return main_parser.parse_args()
