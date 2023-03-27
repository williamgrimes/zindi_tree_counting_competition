"""Zindi Africa palm tree counting competition."""
from core import argparser, efficientnet
import sys

from core.logs import ProjectLogger

logger = ProjectLogger(__name__)

if __name__ == '__main__':
    sys.stdout.write = logger.i  # Redirect stdout to logger to include print statements in log file
    parsed_args = argparser.parse_project_args()

    kwargs = vars(parsed_args)

    command_map = {
        "efficientnet": efficientnet.main,
    }

    logger.setup_info(kwargs, exclude_list=["auth_token"])

    command_map.get(kwargs.get("command"))(kwargs)


