import pandas as pd
import os, sys, pickle, subprocess, shutil
from pathlib import Path

from GANDLF.data.generate_calib_data import generate_calib_data
from GANDLF.utils import get_dataframe, load_model, load_ov_model, optimize_and_save_model
from openvino import runtime as ov
from openvino.tools import mo
from openvino.runtime import serialize, Model
import nncf
from nncf.common.logging.logger import set_log_level

def transform_fn(data_item):
    """
    Extract the model's input from the data item.
    The data item here is the data item that is returned from the data source per iteration.
    This function should be passed when the data item cannot be used as model's input.
    """
    images, _ = data_item
    return images

def PotManager(
    dataframe_calib,
    model_dir,
    outputDir,
    parameters,
):
    """
    This is the training manager that ties all the training functionality together

    Args:
        dataframe_calib (pandas.DataFrame): The calibration data from CSV.
        outputDir (str): The main output directory.
        parameters (dict): The parameters dictionary.
    """

    model, pot_dataloader = generate_calib_data(
        calib_data=dataframe_calib,
        output_dir=outputDir,
        params=parameters,
    )

    calibration_dataset = nncf.Dataset(pot_dataloader, transform_fn)
    
    if os.path.exists(model_dir):
        core = ov.Core()
        model = core.read_model(model_dir)
        #model, _, _ = load_ov_model(model_dir, "CPU")
        '''This will be revisited for PyTorch model NNCF POT quantization
        try:
            main_dict = load_model(model_dir, parameters["device"])
            version_check(paramete["version"], version_to_check=main_dict["version"])
            model.load_state_dict(main_dict["model_state_dict"])
            start_epoch = main_dict["epoch"]
            optimizer.load_state_dict(main_dict["optimizer_state_dict"])
            best_loss = main_dict["loss"]
            print("Previous model successfully loaded.")
        except RuntimeWarning:
            RuntimeWarning("Previous model could not be loaded, initializing model")
        '''
    quantized_model = nncf.quantize(
        model,
        calibration_dataset,
        target_device=nncf.TargetDevice.CPU,
    )

    ov.serialize(quantized_model, os.path.join(outputDir, os.path.basename(model_dir)))
    
