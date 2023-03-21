"""EfficientNet approach"""

from core.logs import ProjectLogger

logger = ProjectLogger(__name__)

def setup_args(subparsers):
    """Argument paser for efficientnet project approach."""

    subparser_efficientnet = subparsers.add_parser("efficientnet")

    subparser_efficientnet.add_argument("--train_split",
                                         default=0.85,
                                         type=float,
                                         help="Ratio of training to test data.")

    subparser_efficientnet.add_argument("--learning_rate",
                                         default=0.001,
                                         type=float,
                                         help="Learning rate.")

    subparser_efficientnet.add_argument("--batch_size",
                                         default=2,
                                         type=int,
                                         help="Batch size for fine tuning.")

    subparser_efficientnet.add_argument("--max_epochs",
                                         default=100,
                                         type=int,
                                         help="Number of epochs.")

    subparser_efficientnet.add_argument("--image_height",
                                         default=1024,
                                         type=int,
                                         help="Height of image.")

    subparser_efficientnet.add_argument("--image_width",
                                         default=1024,
                                         type=int,
                                         help="Width of image.")

    subparser_efficientnet.add_argument("--image_rgb_means",
                                         default=[0.5, 0.5, 0.5],
                                         type=list,
                                         help="RGB image mean normalization.")

    subparser_efficientnet.add_argument("--image_rgb_std",
                                         default=[0.5, 0.5, 0.5],
                                         type=list,
                                         help="RGB standard deviation normalization.")

    subparser_efficientnet.add_argument("--image_max_pixel_value",
                                         default=255,
                                         type=int,
                                         help="Max pixel value for 8-bit images.")

def main(kwargs):
    root_filename = f"{logger.now}_{kwargs['command']}_"
    logger.d(f"{root_filename=}")
    print("you are in efficientnet main")

