"""Zindi Africa palm tree counting competition."""
import sys

from core import argparser, data_downloader, data_normalizer, train
from core.logs import ProjectLogger
from core.utils import set_seed

logger = ProjectLogger(__name__)

if __name__ == '__main__':
    sys.stdout.write = logger.i  # Redirect stdout to logger to include print statements in log file
    parsed_args = argparser.parse_project_args()

    kwargs = vars(parsed_args)

    command_map = {
        "data_downloader": data_downloader.main,
        "data_normalizer": data_normalizer.main,
        "train": train.main,
    }

    logger.setup_info(kwargs, exclude_list=["auth_token"])

    set_seed(kwargs.get("seed"))
    command_map.get(kwargs.get("command"))(kwargs)


