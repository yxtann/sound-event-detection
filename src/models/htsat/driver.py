import os
from loguru import logger
from pathlib import Path


from src.config import DETECTION_TRAIN_PATH, DETECTION_TEST_PATH
from src.utils.audio_to_spectrograms import create_spectrogram_pkl

HTSAT_CHECKPOINT = Path("checkpoints") / "htsat_detector.pth"

def run_htsat(test_files,
    checkpoint_path = HTSAT_CHECKPOINT,
):
    if not os.path.exists(checkpoint_path):
        # TODO
        logger.info("HTS-AT model was trained separately.")
        # logger.info("No model checkpoint, training model...")
        # trained_solver = train_yamnet(checkpoint_path=checkpoint_path)
        # logger.info(f"Saved solver to {checkpoint_path}")
    else:
        logger.info(f"Loading from checkpoint {checkpoint_path}...")


        # trained_solver = Solver(
        #     train_idx=[], checkpoint_path=checkpoint_path, mode="infer"
        # )
        # trained_solver.load_model(checkpoint_path)

    # events_list = trained_solver.inference(test_files)
    events_list = []
    return events_list

