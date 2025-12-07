from pathlib import Path
from datasets import ClassLabel
import torch

from src.config import CLASSES

HTSAT_CHECKPOINT = Path("checkpoints") / "htsat_combined.pth"
HTSAT_CLASSIFICATION_CHECKPOINT = Path("checkpoints") / "htsat_classification.pth"
NON_EVENT_LABEL = 'non_event'
CLASS_LABELS = ClassLabel(names=CLASSES + [NON_EVENT_LABEL])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# loss basis
FRAME_LOSS_BASIS = "FRAME"
CLIP_LOSS_BASIS = "CLIP"
