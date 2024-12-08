import warnings

import hydra
import torch
from hydra.utils import instantiate
from wvmos import get_wvmos

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main()
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
