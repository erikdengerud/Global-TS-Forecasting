import click
import logging
import yaml
from box import Box
import sys
import glob

sys.path.append("")
logger = logging.getLogger(__name__)

from GPTime.config import cfg
from GPTime.source.data_sourcing import source
from GPTime.preprocess.preprocessing import preprocess
from GPTime.model.train import train
from GPTime.model.evaluate import evaluate

tasks = {
    "source": source,
    "preprocess": preprocess,
    "train": train,
    "evaluate": evaluate,
}
logger = logging.getLogger(__name__)


def main(task, task_cfg):
    try:
        tasks[task](task_cfg)
    except:
        logger.error(f"Task {task} failed")
        raise


@click.command()
@click.option("--cfg_path", required=True)
@click.option(
    "--task",
    type=click.Choice(tasks.keys()),
    required=True,
    help="Name of task to execute",
)
def main_cli(task, cfg_path):
    with open(cfg_path, "r") as ymlfile:
        task_cfg = Box(yaml.safe_load(ymlfile))
    main(task, task_cfg)
