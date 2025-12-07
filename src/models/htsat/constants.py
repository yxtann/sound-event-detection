from pathlib import Path
from datasets import ClassLabel
import torch

from src.config import CLASSES

HTSAT_CHECKPOINT = Path("checkpoints") / "htsat_detector.pth"
NON_EVENT_LABEL = 'non_event'
CLASS_LABELS = ClassLabel(names=CLASSES + [NON_EVENT_LABEL])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
