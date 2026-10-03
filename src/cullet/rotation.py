"""
Guess whether a photo is stored rotated by 0, 90, 180 or 270 degrees.

Uses the EfficientNetV2-S orientation classifier of Duarte Barbosa (MIT license), fine-tuned
on 189k upright photos from COCO, TextOCR and others in all 4 rotations, see
https://github.com/duartebarbosadev/deep-image-orientation-detection

The classifier is run on all 4 rotations of each image. Rotating the input by k steps must
shift the prediction by k, so the 4 predictions are combined into one, which is a lot more
reliable than a single one.
"""

import logging
from pathlib import Path

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.hub import download_url_to_file
from torchvision import transforms
from torchvision.models import efficientnet_v2_s

from cullet.paths import get_cache_dir

logger = logging.getLogger(__name__)

MODEL_NAME = "orientation_model_v2_0.9882"
MODEL_URL = (
    "https://huggingface.co/DuarteBarbosa/deep-image-orientation-detection/resolve/main/"
    f"{MODEL_NAME}.pth"
)
# the model was trained on 384x384 center crops of 416x416 resized images
INPUT_SIZE = 384
RESIZE_SIZE = INPUT_SIZE + 32
N_ROTATIONS = 4
# height of the side by side check images
CHECK_HEIGHT = 800
# rotate counter-clockwise by 90 * k degrees, like torch.rot90 on the (height, width) dims
TRANSPOSES = [
    None,
    Image.Transpose.ROTATE_90,
    Image.Transpose.ROTATE_180,
    Image.Transpose.ROTATE_270,
]


def rotate_image(image: Image.Image, k: int) -> Image.Image:
    """Rotate counter-clockwise by 90 * k degrees."""
    k %= N_ROTATIONS
    if k == 0:
        return image
    return image.transpose(TRANSPOSES[k])


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_default_model_file() -> Path:
    return get_cache_dir() / "pretrained_models" / "rotation" / f"{MODEL_NAME}.pth"


def get_rotation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize([RESIZE_SIZE, RESIZE_SIZE]),
            transforms.CenterCrop(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class OrientationClassifier:
    """Classify the rotation of images, for all 4 rotations of each image at once."""

    def __init__(
        self, model_file: Path | None = None, device: str | None = None, batch_size: int = 32
    ):
        if model_file is None:
            model_file = get_default_model_file()
        model_file = Path(model_file)
        if not model_file.is_file():
            logger.info(f"Downloading orientation model from {MODEL_URL} to {model_file}")
            model_file.parent.mkdir(parents=True, exist_ok=True)
            part_file = model_file.with_suffix(".part")
            download_url_to_file(MODEL_URL, part_file.as_posix(), progress=True)
            part_file.replace(model_file)
        if device is None:
            device = get_default_device()
        self.device = torch.device(device)
        self.batch_size = batch_size
        logger.info(f"Loading orientation model {model_file} on {self.device}")
        self.model = efficientnet_v2_s(weights=None)
        n_features = self.model.classifier[1].in_features
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3), torch.nn.Linear(n_features, N_ROTATIONS)
        )
        state_dict = torch.load(model_file, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device).eval()
        self.transform = get_rotation_transform()

    @torch.inference_mode()
    def classify_rotations(self, image_tensors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_tensors: (N, 3, INPUT_SIZE, INPUT_SIZE) preprocessed with self.transform

        Returns:
            (N, 4, 4) log probabilities on cpu. Index k along dim 1 is the image rotated by k
            steps counter-clockwise, index c along dim 2 is the predicted rotation of that view
            in steps counter-clockwise from upright.
        """
        assert image_tensors.ndim == 4, f"Expected (N, 3, H, W), got {image_tensors.shape}"
        rotated = torch.stack(
            [torch.rot90(image_tensors, k, dims=(2, 3)) for k in range(N_ROTATIONS)], dim=1
        )
        flat = rotated.flatten(0, 1)
        log_probs = []
        for start in range(0, len(flat), self.batch_size):
            batch = flat[start : start + self.batch_size].to(self.device)
            log_probs.append(torch.log_softmax(self.model(batch).float(), dim=-1).cpu())
        return torch.cat(log_probs).unflatten(0, (len(image_tensors), N_ROTATIONS))


class RotationImageDataset(torch.utils.data.Dataset):
    """Load images in worker processes. Unreadable images yield an error string."""

    def __init__(self, image_files: list[Path], transform: transforms.Compose):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor | None, int, int, str | None]:
        image_file = self.image_files[idx]
        try:
            with Image.open(image_file) as opened_image:
                # jpeg decoding is much faster when the decoder is told the target size upfront
                opened_image.draft("RGB", (RESIZE_SIZE * 2, RESIZE_SIZE * 2))
                image = ImageOps.exif_transpose(opened_image).convert("RGB")
                # the size as a viewer shows it, before the draft mode shrinks it
                width, height = ImageOps.exif_transpose(opened_image).size
        except (UnidentifiedImageError, OSError) as e:
            return idx, None, 0, 0, f"{type(e).__name__}: {e}"
        except Exception as e:
            raise ValueError(f"Unhandled exception for file {image_file}: {e}") from e
        return idx, self.transform(image), width, height, None


def collate_as_lists(batch):
    return list(zip(*batch))


def make_check_image(
    original: Image.Image, rotated: Image.Image, height: int = CHECK_HEIGHT, gap: int = 20
) -> Image.Image:
    """Put the original on the left and the rotated image on the right, downscaled to the height."""
    panels = []
    for image in (original, rotated):
        if image.height > height:
            image = image.resize((round(image.width * height / image.height), height))
        panels.append(image.convert("RGB"))
    canvas = Image.new(
        "RGB", (panels[0].width + gap + panels[1].width, max(p.height for p in panels)), "white"
    )
    canvas.paste(panels[0], (0, 0))
    canvas.paste(panels[1], (panels[0].width + gap, 0))
    return canvas


def combine_rotation_views(log_probs: torch.Tensor) -> torch.Tensor:
    """
    Combine the predictions for all 4 rotations of each image.

    If the stored image is the upright image rotated by r steps, then rotating it by k more
    steps gives the upright image rotated by r + k, so view k must be classified as r + k.

    Args:
        log_probs: (N, 4, 4) from OrientationClassifier.classify_rotations

    Returns:
        (N, 4) probabilities that the stored image is rotated by r steps counter-clockwise
    """
    scores = torch.zeros(len(log_probs), N_ROTATIONS)
    for k in range(N_ROTATIONS):
        for r in range(N_ROTATIONS):
            scores[:, r] += log_probs[:, k, (r + k) % N_ROTATIONS]
    return torch.softmax(scores, dim=-1)


def evaluate_rotations(log_probs: torch.Tensor, min_confidence: float) -> dict[str, float]:
    """
    Evaluate on log probabilities of upright images, rotating them by each of the 4 steps.

    Args:
        log_probs: (N, 4, 4) of upright images
        min_confidence: combined predictions below it count as unsure

    Returns:
        accuracy of a single view, accuracy of the combined prediction, fraction of confident
        combined predictions and their accuracy
    """
    single_correct = 0
    combined_correct = 0
    confident_correct = 0
    n_confident = 0
    for r in range(N_ROTATIONS):
        # the image rotated by r has the views shifted by r
        rotated = torch.roll(log_probs, shifts=-r, dims=1)
        single_correct += int((rotated[:, 0].argmax(-1) == r).sum())
        probs = combine_rotation_views(rotated)
        correct = probs.argmax(-1) == r
        combined_correct += int(correct.sum())
        confident = probs.max(-1).values >= min_confidence
        n_confident += int(confident.sum())
        confident_correct += int((correct & confident).sum())
    n_total = len(log_probs) * N_ROTATIONS
    return {
        "single_acc": single_correct / n_total,
        "combined_acc": combined_correct / n_total,
        "confident_frac": n_confident / n_total,
        "confident_acc": confident_correct / max(n_confident, 1),
    }
