import os
import soundfile as sf
import shutil
import pickle
import sed_eval
import argparse
import json

from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.config import (
    NUM_STAGES,
    DETECTOR_MODEL,
    CLASSIFIER_MODEL,
    COMBINED_MODEL,
    CLASSES,
    DETECTION_TEST_PATH,
    YAMNET_EXTRACTED_AUDIO_PATH,
)
from src.utils.cut_audio import cut_events_from_audio
from src.utils.generate_ground_truth import generate_gt_events_dict

GT_EVENT_DICT = generate_gt_events_dict()


def calculate_metrics(
    pred_event_dict, gt_event_dict=GT_EVENT_DICT, time_resolution=1.0, t_collar=0.25
):
    # DCASE SED eval: https://tut-arg.github.io/sed_eval/tutorial.html#id1
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
        CLASSES, t_collar=t_collar
    )
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
        CLASSES, time_resolution=time_resolution
    )
    for file, estimated_event in pred_event_dict.items():
        ref_event = gt_event_dict[file]
        event_based_metrics.evaluate(
            reference_event_list=ref_event, estimated_event_list=estimated_event
        )
        segment_based_metrics.evaluate(
            reference_event_list=ref_event, estimated_event_list=estimated_event
        )
    print(event_based_metrics)
    print(segment_based_metrics)


def run_pipeline(args):

    # Combined Model Pipeline
    if args.num_stages == 1:
        if args.combined_model == "yamnet":
            from src.models.single_stage_yamnet_frame import (
                precompute_embeddings,
                run_yamnet_singlestage,
            )

            data_test = pickle.load(
                open(
                    Path("data") / "processed" / "yamnet" / "spectrograms_test.pkl",
                    "rb",
                )
            )
            test_files = [
                os.path.join(DETECTION_TEST_PATH, file) for file in data_test["files"]
            ]
            precompute_embeddings(test_files)
            events_list = run_yamnet_singlestage(test_files)
        elif args.combined_model == "crnn":
            from src.models.crnn.crnn import run_crnn
            events_list = run_crnn(create_data=False, retrain=True)
        elif args.combined_model == "htsat":
            from src.models.htsat.combined import run_htsat_combined

            events_list = run_htsat_combined()
        else:
            raise Exception(
                f"Invalid COMBINED_MODEL {args.combined_model} for {args.num_stages} stage pipeline"
            )

        assert (
            events_list
        ), "Error: The combined model did not return any events (events_list is empty)."

    # Two-Stage Pipeline: Detection then Classification
    if args.num_stages == 2:

        # Detector Stage
        if args.detector_model == "yamnet":
            from src.models.yamnet_train import run_yamnet

            test_path = Path("data") / "processed" / "yamnet" / "spectrograms_test.pkl"
            test_data = pickle.load(open(test_path, "rb"))
            filepaths = [
                os.path.join(DETECTION_TEST_PATH, file) for file in test_data["files"]
            ]  # filepath of test wav files
            events_list = run_yamnet(
                filepaths,
                checkpoint_path="checkpoints/yamnet_detector.pth",
            )
        else:
            raise Exception(
                f"Invalid DETECTOR_MODEL {args.detector_model} for {args.num_stages} stage pipeline"
            )

        assert (
            events_list
        ), "Error: The detector model did not return any events (events_list is empty)."

        # Classifier Stage
        if args.classifier_model == "yamnet":
            pass

        elif args.classifier_model == "crnn":
            pass

        elif args.classifier_model == "htsat":
            from src.models.htsat.classification import run_htsat_classification

            updated_events_list = cut_events_from_audio(
                YAMNET_EXTRACTED_AUDIO_PATH, events_list
            )

            events_list = run_htsat_classification(updated_events_list)

        elif args.classifier_model == "mamba":
            # Lazy loading for mamba
            from src.models.audio_mamba_ft import audio_mamba_inference
            from src.utils.audio_mamba_metadata_generator import (
                generate_metadata_from_detector,
            )

            events_list = cut_events_from_audio(
                YAMNET_EXTRACTED_AUDIO_PATH, events_list
            )

            generate_metadata_from_detector(
                groundtruth_csv=os.path.join(DETECTION_TEST_PATH, "annotations.csv"),
                detector_audio_output_path=YAMNET_EXTRACTED_AUDIO_PATH,
            )

            path_pred_dict = audio_mamba_inference(
                checkpoint_path="checkpoints/audio_mamba_ft.pth",
                val_json_path=os.path.join(
                    YAMNET_EXTRACTED_AUDIO_PATH, "audio_mamba_metadata.json"
                ),
            )

            # Add the label
            all_count = 0
            corr_count = 0
            for audio_file in events_list:
                for event in events_list[audio_file]:
                    all_count += 1
                    try:
                        extracted_audio_filename = event["extracted_audio_filename"]
                        event["event_label"] = path_pred_dict[extracted_audio_filename][
                            "pred_class"
                        ]
                        if event["ground_truth"] != event["event_label"]:
                            event["match"] = False
                        else:
                            event["match"] = True
                            corr_count += 1
                    except:
                        # This shouldn't happen as everything should have a prediction
                        raise Exception(
                            f"Error: Extracted audio filename {extracted_audio_filename} not found in path_pred_dict"
                        )

            print(f"All count: {all_count}")
            print(f"Correct count: {corr_count}")
            print(f"Accuracy: {corr_count / all_count}")

        else:
            raise Exception(
                f"Invalid CLASSIFIER_MODEL {args.classifier_model} for {args.num_stages} stage pipeline"
            )

    # Run metrics
    gt_events_dict = generate_gt_events_dict()

    # Save the event dict
    if args.num_stages == 2:
        with open(
            f"events_list_results/events_list_stage_{args.num_stages}_{args.detector_model}_{args.classifier_model}.json",
            "w",
        ) as f:
            json.dump(events_list, f, indent=4)
    else:
        with open(
            f"events_list_results/events_list_stage_{args.num_stages}_{args.combined_model}.json",
            "w",
        ) as f:
            json.dump(events_list, f, indent=4)

    calculate_metrics(events_list, gt_events_dict)


"""
Example Usage: python main.py --num-stages=2 --detector-model=yamnet --classifier-model=htsat
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Audio Detection Pipeline")

    parser.add_argument(
        "--num-stages", type=int, default=NUM_STAGES, help="Optional: Number of stages"
    )

    parser.add_argument(
        "--detector-model",
        type=str,
        default=DETECTOR_MODEL,
        help="Optional: Detector Model",
    )

    parser.add_argument(
        "--classifier-model",
        type=str,
        default=CLASSIFIER_MODEL,
        help="Optional: Classifier Model",
    )

    parser.add_argument(
        "--combined-model",
        type=str,
        default=CLASSIFIER_MODEL,
        help="Optional: Combined Model",
    )

    args = parser.parse_args()

    print(args)

    run_pipeline(args)
