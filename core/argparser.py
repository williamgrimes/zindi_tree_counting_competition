"""Argument paser for project"""
import argparse
from core import data_downloader, data_normalizer, efficientnet

from pathlib import Path


def files_args(main_parser, args):
    main_parser.add_argument("--seed",
                             default=42,
                             type=int,
                             help="Pseudo-random number generator seed.",
                             )
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
                             type=Path,
                             help="Path to test data csv.")
    main_parser.add_argument("--sample_submission_csv",
                             default=Path(args.data_dir, "SampleSubmission.csv"),
                             type=Path,
                             help="Path to sample submission csv.")
    main_parser.add_argument("--runs_csv",
                             default=Path(args.runs_dir, "runs.csv"),
                             type=Path,
                             help="Path to csv containing run_name and loss.")
    return main_parser


def parse_project_args():
    """Argument paser for project"""
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--params_file",
                               default="params.yaml",
                               type=str,
                               help="Folder containing run aprameters.")
    parent_parser.add_argument("--data_dir",
                               default="data",
                               type=str,
                               help="Folder containing project data.")
    parent_parser.add_argument("--logs_dir",
                               default="logs",
                               type=str,
                               help="Folder containing project logs.")
    parent_parser.add_argument("--runs_dir",
                               default="runs",
                               type=str,
                               help="Folder containing training run experiments.")

    subparsers = parent_parser.add_subparsers(dest="command", required=True)
    data_downloader.setup_args(subparsers)
    data_normalizer.setup_args(subparsers)
    efficientnet.setup_args(subparsers)

    main_parser = argparse.ArgumentParser(parents=[parent_parser])

    main_parser = files_args(main_parser, parent_parser.parse_args())

    return main_parser.parse_args()
