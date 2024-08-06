import os
import torch
from tqdm import tqdm
import torchio

from GANDLF.utils import (
    print_model_summary,
    parseTrainingCSV
)
from GANDLF.logger import Logger
from GANDLF.compute.generic import create_pytorch_objects
from GANDLF.utils import (
    get_date_time,
    populate_channel_keys_in_params,
)
from torch.utils.data import (
    TensorDataset,
    DataLoader,
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
        mode="calib",
    )

    # Defining our model here according to parameters mentioned in the configuration file
    print("Number of channels : ", params["model"]["num_channels"])

    (
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
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

    #calib_logger.write_header(mode="calib")

    print("Using device:", device, flush=True)

    calib_image_dataset = []
    calib_label_dataset = []

    for batch_idx, (subject) in enumerate(
        tqdm(train_dataloader, desc="Looping over training data")
    ):
        image = (  # 5D tensor: (B, C, H, W, D)
            torch.cat(
                [subject[key][torchio.DATA] for key in params["channel_keys"]], dim=1
            )
            .float()
            .to(params["device"])
        )
        if (
            "value_keys" in params
        ):  # classification / regression (when label is scalar) or multilabel classif/regression
            label = torch.cat([subject[key] for key in params["value_keys"]], dim=0)
            # min is needed because for certain cases, batch size becomes smaller than the total remaining labels
            label = label.reshape(
                min(params["batch_size"], len(label)), len(params["value_keys"])
            )
        else:
            label = subject["label"][
                torchio.DATA
            ]  # segmentation; label is (B, C, H, W, D) image
        label = label.to(params["device"])

        if params["save_training"]:
            write_training_patches(subject, params)

        # ensure spacing is always present in params and is always subject-specific
        if "spacing" in subject:
            params["subject_spacing"] = subject["spacing"]
        else:
            params["subject_spacing"] = None
        
        calib_image_dataset.append(image.squeeze())
        calib_label_dataset.append(label.squeeze())


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

    calib_image_sample = torch.squeeze(torch.Tensor([calib_image_dataset[i].numpy() for i in index]))
    calib_label_sample = torch.squeeze(torch.Tensor([calib_label_dataset[i].numpy() for i in index]))
    print(calib_image_sample.shape)

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
