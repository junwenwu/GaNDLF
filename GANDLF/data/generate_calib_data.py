import os, time, psutil
import torch
from tqdm import tqdm
import numpy as np
import torchio

from GANDLF.utils import (
    print_model_summary,
)
from GANDLF.logger import Logger
from GANDLF.compute.generic import create_pytorch_objects
from GANDLF.utils import (
    populate_channel_keys_in_params,
    get_ground_truths_and_predictions_tensor,
)
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    random_split,
    SubsetRandomSampler,
)
import random

# hides torchio citation request, see https://github.com/fepegar/torchio/issues/235
os.environ["TORCHIO_HIDE_CITATION_PROMPT"] = "1"


def generate_calib_data(
    calib_data,
    params,
    output_dir,
):
    """
    The main calibration.

    Args:
        calib_data (pandas.DataFrame): The data to use for calibration.
        params (dict): The parameters dictionary.
        output_dir (str): The output directory.
    """
    # Some autodetermined factors
    device = "cpu"
    params["device"] = device
    params["output_dir"] = output_dir
    params["training_data"] = calib_data

    if not ("weights" in params):
        params["weights"] = None  # no need for loss weights for inference
    if not ("class_weights" in params):
        params["class_weights"] = None  # no need for class weights for inference

    calib_logger = Logger(
        logger_csv_filename=os.path.join(output_dir, "logs_calib.csv"),
        metrics=params["metrics"],
    )

    # Defining our model here according to parameters mentioned in the configuration file
    print("Number of channels : ", params["model"]["num_channels"])

    (
        model,
        optimizer,
        train_dataloader,
        _,
        scheduler,
        params,
    ) = create_pytorch_objects(params, calib_data, None, device)

    params = populate_channel_keys_in_params(train_dataloader, params)

    if params["model"]["print_summary"]:
        print_model_summary(
            model,
            params["batch_size"],
            params["model"]["num_channels"],
            params["patch_size"],
            params["device"],
        )

    calib_logger.write_header(mode="calib")

    print("Using device:", device, flush=True)

    calib_image_dataset = []
    calib_label_dataset = []

    for batch_idx, (subject) in enumerate(
        tqdm(train_dataloader, desc="Looping over training data")
    ):
        # ensure spacing is always present in params and is always subject-specific
        params["subject_spacing"] = None
        if "spacing" in subject:
            params["subject_spacing"] = subject["spacing"]

        # constructing a new dict because torchio.GridSampler requires torchio.Subject,
        # which requires torchio.Image to be present in initial dict, which the loader does not provide
        subject_dict = {}
        label_ground_truth = None
        label_present = False
        # this is when we want the dataloader to pick up properties of GaNDLF's DataLoader, such as pre-processing and augmentations, if appropriate
        if "label" in subject:
            if subject["label"] != ["NA"]:
                subject_dict["label"] = torchio.Image(
                    path=subject["label"]["path"],
                    type=torchio.LABEL,
                    tensor=subject["label"]["data"].squeeze(0),
                    affine=subject["label"]["affine"].squeeze(0),
                )
                label_present = True
                label_ground_truth = subject_dict["label"]["data"]
        if "value_keys" in params:  # for regression/classification
            for key in params["value_keys"]:
                subject_dict["value_" + key] = subject[key]
                label_ground_truth = torch.cat(
                    [subject[key] for key in params["value_keys"]], dim=0
                )

        for key in params["channel_keys"]:
            subject_dict[key] = torchio.Image(
                path=subject[key]["path"],
                type=subject[key]["type"],
                tensor=subject[key]["data"].squeeze(0),
                affine=subject[key]["affine"].squeeze(0),
            )

        # regression/classification problem AND label is present
        if (params["problem_type"] != "segmentation") and label_present:
            sampler = torchio.data.LabelSampler(params["patch_size"])
            tio_subject = torchio.Subject(subject_dict)
            generator = sampler(tio_subject, num_patches=params["q_samples_per_volume"])
            for patch in generator:
                image = torch.cat(
                    [patch[key][torchio.DATA] for key in params["channel_keys"]], dim=0
                )
                valuesToPredict = torch.cat(
                    [patch["value_" + key] for key in params["value_keys"]], dim=0
                )
                image = image.unsqueeze(0)
                image = image.float().to(params["device"])
                ## special case for 2D
                if image.shape[-1] == 1:
                    image = torch.squeeze(image, -1)

                calib_image_dataset.append(image)
                calib_label_dataset.append(valuesToPredict)

        else:  # for segmentation problems OR regression/classification when no label is present
            grid_sampler = torchio.inference.GridSampler(
                torchio.Subject(subject_dict),
                params["patch_size"],
                patch_overlap=params["inference_mechanism"]["patch_overlap"],
            )
            patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1)
            aggregator = torchio.inference.GridAggregator(
                grid_sampler,
                overlap_mode=params["inference_mechanism"]["grid_aggregator_overlap"],
            )

            if params["medcam_enabled"]:
                attention_map_aggregator = torchio.inference.GridAggregator(
                    grid_sampler,
                    overlap_mode=params["inference_mechanism"][
                        "grid_aggregator_overlap"
                    ],
                )

            current_patch = 0
            for patches_batch in patch_loader:
                if params["verbose"]:
                    print(
                        "=== Current patch:",
                        current_patch,
                        ", time : ",
                        get_date_time(),
                        ", location :",
                        patches_batch[torchio.LOCATION],
                        flush=True,
                    )
                current_patch += 1
                image = (
                    torch.cat(
                        [
                            patches_batch[key][torchio.DATA]
                            for key in params["channel_keys"]
                        ],
                        dim=1,
                    )
                    .float()
                    .to(params["device"])
                )

                # calculate metrics if ground truth is present
                label = None
                if params["problem_type"] != "segmentation":
                    label = label_ground_truth
                elif "label" in patches_batch:
                    label = patches_batch["label"][torchio.DATA]

                if label is not None:
                    label = label.to(params["device"])
                    if params["verbose"]:
                        print(
                            "=== Calibration shapes : label:",
                            label.shape,
                            ", image:",
                            image.shape,
                            flush=True,
                        )
                calib_image_dataset.append(image)
                calib_label_dataset.append(label)

    random.seed(params["calib_data_sample_seed"])
    if "calib_sample_size" not in params.keys():
        params["calib_sample_size"] = 300

    if float(params["calib_sample_size"]) > 1:
        index = random.sample(
            range(len(calib_image_dataset)),
            int(params["calib_sample_size"]),
        )
    else:
        index = random.sample(
            range(len(calib_image_dataset)),
            int(params["calib_sample_size"] * len(calib_image_dataset)),
        )

    calib_image_sample = torch.Tensor([calib_image_dataset[i].numpy() for i in index])
    calib_label_sample = torch.Tensor([calib_label_dataset[i].numpy() for i in index])

    calib_dataset = TensorDataset(calib_image_sample, calib_label_sample)

    calib_dataloader = DataLoader(dataset=calib_dataset, shuffle=False, batch_size=1)
    return (model, calib_dataloader)


if __name__ == "__main__":
    import argparse, pickle, pandas

    torch.multiprocessing.freeze_support()
    # parse the cli arguments here
    parser = argparse.ArgumentParser(description="NNCF PTQ of Trained Model")
    parser.add_argument(
        "-c",
        "--calib_data_csv",
        type=str,
        help="CSV file of the calibration data",
        required=True,
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, help="Output directory", required=True
    )
    parser.add_argument(
        "-p", "--parameter_pickle", type=str, help="Parameters pickle", required=True
    )
    args = parser.parse_args()

    # # write parameters to pickle - this should not change for the different folds, so keeping is independent
    parameters = pickle.load(open(args.parameter_pickle, "rb"))
    calib_data, headers = parseTrainingCSV(args.calib_data_csv, train=True)

    model, ptq_dataloader = generate_calib_data(
        calib_data=calib_data,
        output_dir=args.output_dir,
        params=parameters,
    )
