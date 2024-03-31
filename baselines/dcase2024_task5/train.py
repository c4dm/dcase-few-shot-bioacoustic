import torch

torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.enabled = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

import dotenv
import hydra
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="train.yaml")
def main(config: DictConfig):
    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.training_pipeline import train

    if config.disable_cudnn:
        print("cudnn is disabled, the trianing may be slower")
        torch.backends.cudnn.enabled = False
    # Applies optional utilities
    utils.extras(config)

    # Train model
    return train(config)


if __name__ == "__main__":
    main()
