"""
SSCD image copy detection descriptor (Pizzi et al. 2022) for finding duplicate images.

The torchscript model runs without any SSCD code, see
https://github.com/facebookresearch/sscd-copy-detection
"""

import logging
from pathlib import Path

import torch
from PIL import Image
from torch.hub import download_url_to_file
from torchvision import transforms

from cullet.paths import get_cache_dir

logger = logging.getLogger(__name__)

SSCD_MODEL_NAME = "sscd_disc_large"
SSCD_URL = f"https://dl.fbaipublicfiles.com/sscd-copy-detection/{SSCD_MODEL_NAME}.torchscript.pt"
# the model skews all images to a square, so duplicates with different aspect ratios still match
SSCD_INPUT_SIZE = 320
SSCD_EMBEDDING_DIM = 1024


def get_default_sscd_model_file() -> Path:
    return get_cache_dir() / "pretrained_models" / "sscd" / f"{SSCD_MODEL_NAME}.torchscript.pt"


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_sscd_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize([SSCD_INPUT_SIZE, SSCD_INPUT_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class SscdEmbedder:
    def __init__(
        self, model_file: Path | None = None, device: str | None = None, batch_size: int = 32
    ):
        if model_file is None:
            model_file = get_default_sscd_model_file()
        model_file = Path(model_file)
        if not model_file.is_file():
            logger.info(f"Downloading SSCD model from {SSCD_URL} to {model_file}")
            model_file.parent.mkdir(parents=True, exist_ok=True)
            part_file = model_file.with_suffix(".part")
            download_url_to_file(SSCD_URL, part_file.as_posix(), progress=True)
            part_file.replace(model_file)
        if device is None:
            device = get_default_device()
        self.device = torch.device(device)
        self.batch_size = batch_size
        logger.info(f"Loading SSCD model {model_file} on {self.device}")
        self.model = torch.jit.load(model_file.as_posix(), map_location="cpu").to(self.device)
        self.model.eval()
        self.transform = get_sscd_transform()

    @torch.inference_mode()
    def embed_tensors(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tensors: (N, 3, 320, 320) images preprocessed with get_sscd_transform

        Returns:
            (N, 1024) L2-normalized embeddings on cpu, so the dot product is the cosine similarity
        """
        assert image_tensors.ndim == 4, f"Expected (N, 3, H, W), got {image_tensors.shape}"
        embeddings = []
        for start in range(0, len(image_tensors), self.batch_size):
            batch = image_tensors[start : start + self.batch_size].to(self.device)
            embeddings.append(torch.nn.functional.normalize(self.model(batch), dim=-1).cpu())
        return torch.cat(embeddings)

    def embed_pil_images(self, images: list[Image.Image]) -> torch.Tensor:
        tensors = torch.stack([self.transform(image.convert("RGB")) for image in images])
        return self.embed_tensors(tensors)
