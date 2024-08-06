import os
from pathlib import Path
from typing import Optional
from GANDLF.compute import create_pytorch_objects
from GANDLF.config_manager import ConfigManager
from GANDLF.utils import version_check, load_model, optimize_and_save_model
from GANDLF.data.generate_calib_data import generate_calib_data

from GANDLF.utils import (
    populate_header_in_parameters,
    parseTrainingCSV,
    send_model_to_device,
    get_class_imbalance_weights,
)

import nncf

def transform_fn(data_item):
    """
    Extract the model's input from the data item.
    The data item here is the data item that is returned from the data source per iteration.
    This function should be passed when the data item cannot be used as model's input.
    """
    images, _ = data_item
    return images

def post_training_model_optimization(
    model_path: str, config_path: Optional[str] = None, data_path: Optional[str] = None, output_path: Optional[str] = None
) -> bool:
    """
    CLI function to optimize a model for deployment.

    Args:
        model_path (str): Path to the model file.
        config_path (str, optional): Path to the configuration file.
        data_path (str, optional): Path to the .csv file for the calibration.
        output_path (str, optional): Output directory to save the optimized model.

    Returns:
        bool: True if successful, False otherwise.
    """
    # Load the model and its parameters from the given paths
    main_dict = load_model(model_path, "cpu")
    parameters = main_dict.get("parameters", None)

    # If parameters are not available in the model file, parse them from the config file
    parameters = (
        ConfigManager(config_path, version_check_flag=False)
        if parameters is None
        else parameters
    )


    output_path = os.path.dirname(model_path) if output_path is None else output_path
    Path(output_path).mkdir(parents=True, exist_ok=True)
    optimized_model_path = os.path.join(
        output_path, os.path.basename(model_path).replace("pth.tar", "onnx")
    )

    # Create PyTorch objects and set onnx_export to True for optimization
    # model, _, train_dataloader, val_dataloader, _, parameters = create_pytorch_objects(parameters, train_csv=data_path,  device="cpu")
    parameters["model"]["onnx_export"] = True
    
    train_data_df, headers_to_populate_train = parseTrainingCSV(
            data_path, train=True
    )
    parameters = populate_header_in_parameters(
            parameters, headers_to_populate_train
    )
    
    model, calib_dataloader = generate_calib_data(
        calib_data=data_path,
        output_dir=output_path,
        params=parameters,
    )

    parameters["model"]["onnx_export"] = True
    
    calibration_dataset = nncf.Dataset(calib_dataloader, transform_fn)
    
    # Perform version check and load the model's state dictionary
    #version_check(parameters["version"], version_to_check=main_dict["version"])
    model.load_state_dict(main_dict["model_state_dict"])

    # Optimize the model and save it to an ONNX file
    optimize_and_save_model(model, parameters, optimized_model_path, onnx_export=True)
    
    quantized_model = nncf.quantize(model, calibration_dataset)

    # Check if the optimized model file exists
    if not os.path.exists(optimized_model_path):
        print("Error while optimizing the model.")
        return False

    return True
