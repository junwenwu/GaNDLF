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
from GANDLF.utils import populate_channel_keys_in_params
from torch.utils.data import TensorDataset, DataLoader, random_split, SubsetRandomSampler
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

        image = torch.squeeze(torch.squeeze(torch.cat(
                [subject[key][torchio.DATA] for key in params["channel_keys"]], dim=1
                ).float().to(params["device"]), 0), 3)

        if "value_keys" in params:
            label = torch.cat([subject[key] for key in params["value_keys"]], dim=0)
            # min is needed because for certain cases, batch size becomes smaller than the total remaining labels
            label = label.reshape(
                min(params["batch_size"], len(label)),
                len(params["value_keys"]),
            )
        else:
            label = subject["label"][torchio.DATA]

        label = label.to(params["device"])
        calib_image_dataset.append(image)
        calib_label_dataset.append(label)

    import random
    index = random.sample(range(len(calib_image_dataset)), int(params['calib_ratio'] * len(calib_image_dataset)))
    
    calib_image_sample = torch.Tensor([calib_image_dataset[i].numpy() for i in index])
    calib_label_sample = torch.Tensor([calib_label_dataset[i].numpy() for i in index])

    calib_dataset = TensorDataset(calib_image_sample, calib_label_sample)
    
    calib_dataloader =  DataLoader(dataset = calib_dataset, shuffle=False, batch_size=1)
    return(model, calib_dataloader)


if __name__ == "__main__":
    import argparse, pickle, pandas

    torch.multiprocessing.freeze_support()
    # parse the cli arguments here
    parser = argparse.ArgumentParser(description="NNCF POT of Trained Model")
    parser.add_argument(
        "-calib_data_csv", type=str, help="CSV file of the calibration data", required=True
    )
    parser.add_argument("-output_dir", type=str, help="Output directory", required=True)
    parser.add_argument(
        "-params", type=str, help="Parameters pickle", required=True
    )
    args = parser.parse_args()

    # # write parameters to pickle - this should not change for the different folds, so keeping is independent
    parameters = pickle.load(open(args.parameter_pickle, "rb"))
    calib_data, headers = parseTrainingCSV(args.calib_data_csv, train=True)

    model, pot_dataloader = generate_calib_data(
        calib_data=calib_data,
        output_dir=args.output_dir,
        params=parameters,
    )
