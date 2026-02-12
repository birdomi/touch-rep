import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tactile_ssl.utils.logging import get_pylogger

logging = get_pylogger(__name__)


def build_encoder(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    **kwargs,
):
    # Read config and init model
    with initialize(version_base=None, config_path="..", job_name="build_encoder"):
        cfg = compose(config_name=config_file)
        OmegaConf.resolve(cfg)
        print(OmegaConf.to_yaml(cfg))
    model = instantiate(cfg.encoder, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model


def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=True)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
        torch.save(model.state_dict(), "encoder_sparshskin.pth")
