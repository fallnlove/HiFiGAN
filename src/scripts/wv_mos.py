import warnings

import hydra
import torch
from hydra.utils import instantiate
from wvmos import get_wvmos

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None)
def main(config):
    """
    Main script for calculate WV-MOS.

    Args:
        config (DictConfig): hydra experiment config.
    """
    model = get_wvmos(cuda=torch.cuda.is_available())

    print("WV-MOS", model.calculate_dir(config.data_dir, mean=True))


if __name__ == "__main__":
    main()
