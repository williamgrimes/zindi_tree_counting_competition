"""Zindi Africa palm tree counting competition."""
from core import argparser, efficientnet
from core.logs import ProjectLogger

logger = ProjectLogger(__name__)

if __name__ == '__main__':
    parsed_args = argparser.parse_project_args()

    kwargs = vars(parsed_args)

    command_map = {
        "efficientnet": efficientnet.main,
    }

    logger.setup_info(kwargs)

    command_map.get(parsed_args.command)(kwargs)


