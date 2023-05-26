import os, pickle
from pathlib import Path

from GANDLF.ptq_manager import PtqManager
from GANDLF.parseConfig import parseConfig
from GANDLF.utils import (
    populate_header_in_parameters,
    parseTrainingCSV,
)

import numpy as np


def ptq_run(
    data_csv, config_file, model_dir, output_dir=None
):
    """
    Main function that runs the training and inference.

    Args:
        data_csv (str): The CSV file of the calibration data.
        config_file (str): The YAML file of the training configuration.
        model_dir (str): The model directory for the trained model.
        output_dir (str): The output directory for the NNCF ptq session.

    Returns:
        None
    """
    file_data_full = data_csv

    currentModelConfigPickle = os.path.join(output_dir, "parameters.pkl")

    if (not os.path.exists(currentModelConfigPickle)):
        model_parameters = config_file
        device = "CPU"
        parameters = parseConfig(model_parameters)
        parameters["device"] = "CPU"
        parameters["device_id"] = -1
        
        with open(currentModelConfigPickle, "wb") as handle:
            pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        if os.path.exists(currentModelConfigPickle):
            print(
                "Using previously saved parameter file",
                currentModelConfigPickle,
                flush=True,
            )
            parameters = pickle.load(open(currentModelConfigPickle, "rb"))

    # if the output directory is not specified, then use the model directory even for the testing data
    # default behavior
    parameters["output_dir"] = output_dir
    if output_dir is None:
        parameters["output_dir"] = model_dir
    Path(parameters["output_dir"]).mkdir(parents=True, exist_ok=True)

    # parse training CSV
    data_full, headers = parseTrainingCSV(file_data_full, train=True)
    parameters = populate_header_in_parameters(parameters, headers)
    PtqManager(
                dataframe_calib=data_full,
                model_dir = model_dir, 
                outputDir=parameters["output_dir"],
                parameters=parameters,
    )
