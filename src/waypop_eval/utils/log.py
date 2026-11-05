import logging
import logging.config
import os
import sys
import warnings
from pathlib import Path

import click
import colorlog
import rich
import rich.traceback
import torch
from rich.logging import RichHandler


def initialize() -> None:
    """Initialize logging system."""

    # if the log is being redirected, default to a 120 column output
    if not sys.stdout.isatty() and "COLUMNS" not in os.environ:
        os.environ["COLUMNS"] = "120"

    # set up traceback formatting
    tracebacks_suppress = [
        click,
        rich,
        torch,
    ]

    rich_tracebacks = os.environ.get("RICH_TRACEBACKS", "1") == "1"
    if rich_tracebacks:
        rich.traceback.install(
            show_locals=True,
            width=120,
            suppress=tracebacks_suppress,
        )

    # set up logging
    rich_logs = sys.stdout.isatty()
    rich_logs = os.environ.get("RICH_LOGS", str(int(rich_logs))) == "1"

    if rich_logs:
        handler = RichHandler(
            log_time_format="[%Y-%m-%d %H:%M:%S]",
            omit_repeated_times=False,
            enable_link_path=False,
            rich_tracebacks=rich_tracebacks,
            tracebacks_show_locals=True,
            tracebacks_suppress=tracebacks_suppress,
        )
    else:
        formatter = colorlog.ColoredFormatter(
            "[%(asctime)s][%(bold)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s]: "
            + "%(message)s"
        )

        handler = colorlog.StreamHandler()
        handler.setFormatter(formatter)

    logging.root.setLevel(logging.INFO)
    logging.root.addHandler(handler)

    warnings.showwarning = log_warning

    # fix torch and lightning loggers...
    #
    # For some reason, torch and lightning set up logging handlers
    # themselves... which is annoying as it prevents our config above from
    # taking hold. Meaning torch and lightning logging output is not caught by
    # the root logger and thus not formatted properly (i.e., not formatted like
    # configured above). So for all torch and lightning loggers: Remove all
    # handlers and set the logger to propagate messages upwards.
    for name, logger in logging.Logger.manager.loggerDict.items():
        if name.startswith("lightning.") or name == "lightning":
            if not isinstance(logger, logging.PlaceHolder):
                logger.handlers.clear()
                logger.propagate = True

        if name.startswith("torch.") or name == "torch":
            if not isinstance(logger, logging.PlaceHolder):
                logger.handlers.clear()
                logger.propagate = True

    # demote third-party loggers
    logging.getLogger("aim").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.INFO)
    logging.getLogger("fsspec").setLevel(logging.INFO)
    logging.getLogger("git").setLevel(logging.INFO)
    logging.getLogger("hydra").setLevel(logging.INFO)
    logging.getLogger("shapely").setLevel(logging.INFO)

    # hide some unnecessary warnings

    # - We create the log directory before the CSVLogger can create it itself,
    #   so it complains that it's not empty. That's okay. Shut it up.
    warnings.filterwarnings(
        action="ignore",
        message=".* logs directory .* exists and is not empty. .*",
        module="lightning.fabric.loggers.csv_logs",
    )
    # - Similarly, we create the log directory before the checkpoint code can
    #   create it itself, so it complains that it's not empty. That's okay.
    #   Shut it up.
    warnings.filterwarnings(
        action="ignore",
        message="Checkpoint directory .* exists and is not empty.",
        module="lightning.pytorch.callbacks.model_checkpoint",
    )
    # - To handle dictionaries returned by torchmetric metrics, we manually
    #   need to compute() them. This already takes care of synchronization and
    #   accumulation across devices. Therefore, calling
    #
    #       LightningModule.log(..., sync_dist=True)
    #
    #   is completely unnecessary. So ignore the warning about it.
    warnings.filterwarnings(
        action="ignore",
        # pylint: disable-next=line-too-long
        message="It is recommended to use .* when logging on epoch level in distributed setting to accumulate the metric across devices.",
        module="lightning.pytorch.trainer.connectors.logger_connector.result",
    )
    # - PyTorch uses torch.cpu.amp.autocast() internally (at
    #   torch/utils/checkpoint.py), which is deprecated. Ignore it, that's not
    #   on us to fix...
    warnings.filterwarnings(
        action="ignore",
        message=r"`torch\.cpu\.amp\.autocast\(args\.\.\.\)` is deprecated\. .*",
        module="torch.utils.checkpoint",
        category=FutureWarning,
    )


def add_file_handler(path: Path) -> None:
    formatter = logging.Formatter("[%(asctime)s %(name)s %(levelname)s]: %(message)s")

    fh = logging.FileHandler(path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logging.root.addHandler(fh)


def get_logger(name: str | None = None):
    return logging.getLogger(name)


def _get_logger_for_path(path: Path | str) -> logging.Logger:
    """
    Retrieve or create a logger based on a file path.

    This function attempts to find a logger corresponding to the module associated
    with the given file path. If no module is found, it defaults to using the
    base name (stem) of the file as the logger name.

    Args:
        path (Union[Path, str]): The path to the file or module.

    Returns:
        logging.Logger: A logger instance corresponding to the file or module.
    """
    # Ensure the path is resolved to an absolute Path object
    path = Path(path).resolve()

    # Attempt to find the module name by its path
    for module_name, module in sys.modules.items():
        module_path = getattr(module, "__file__", None)
        if module_path and Path(module_path).resolve() == path:
            break
    else:
        # Default to the stem of the path if no module is found
        module_name = path.stem

    # Return a logger named after the module or file
    return logging.getLogger(module_name)


def log_warning(
    message: str,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: object | None = None,
    line: str | None = None,
) -> None:
    # pylint: disable=unused-argument
    """
    Log a formatted warning message using a logger based on the file path.

    This function formats the warning message and logs it using a logger
    retrieved based on the file path where the warning originated.

    Intended to replace warnings.showwarning.

    Args:
        message (str): The warning message.
        category (type[Warning]): The warning category (e.g., UserWarning, DeprecationWarning).
        filename (str): The path to the file where the warning was raised.
        lineno (int): The line number in the file where the warning was raised.
        file (Optional[object]): Unused in this implementation.
        line (Optional[str]): The line of code triggering the warning (optional).

    Returns:
        None
    """
    # Format the warning message
    message = warnings.formatwarning(message, category, filename, lineno, line)
    message = message.strip("\n")

    # Retrieve the logger for the given filename and log the warning
    log = _get_logger_for_path(filename)
    log.warning(message)
