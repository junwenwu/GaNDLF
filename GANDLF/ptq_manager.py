import pandas as pd
import os, sys, pickle, subprocess, shutil
from pathlib import Path

from GANDLF.data.generate_calib_data import generate_calib_data
from GANDLF.utils import get_dataframe, load_model, load_ov_model, optimize_and_save_model, get_ground_truths_and_predictions_tensor
from openvino import runtime as ov
from openvino.tools import mo
from openvino.runtime import serialize, Model, CompiledModel
import nncf
from nncf.common.logging.logger import set_log_level
import torch
import numpy as np
from GANDLF.compute.loss_and_metric import get_loss_and_metrics

def transform_fn(data_item):
    """
    Extract the model's input from the data item.
    The data item here is the data item that is returned from the data source per iteration.
    This function should be passed when the data item cannot be used as model's input.
    """
    images, _ = data_item
    return images

def getMetricsParams(parameters):
    global accuracy_parameters
    accuracy_parameters = parameters

def validate(model: CompiledModel,
             validation_loader: torch.utils.data.DataLoader) -> float:

    total_calib_metric = 0
    average_calib_metric = 0

    input_layer = model.inputs
    output_layer = model.outputs


    calculate_overall_metrics = (
        (accuracy_parameters["problem_type"] == "classification")
        or (accuracy_parameters["problem_type"] == "regression")
    ) 

    # accuracy_parameters["weights"] = None
    count = 0
    for image, target in validation_loader:
        count += 1
        image = np.expand_dims(torch.squeeze(image).cpu().numpy(), axis=0)
        pred_output = torch.from_numpy(
            model(inputs={input_layer[0]: image})[output_layer[0]]
        )
        pred_output = pred_output.to(accuracy_parameters["device"])       

        if accuracy_parameters["problem_type"] == "regression":
            output = torch.FloatTensor([np.argmax(np.asarray(pred_output[0]), 0)])
        elif accuracy_parameters["problem_type"] == "classification":
            output = pred_output.to(torch.float)
            target = target.to(torch.long)
        else:
            output = pred_output.to(torch.float)

        # for the weird cases where mask is read as an RGB image, ensure only the first channel is used
        if target is not None:
            if accuracy_parameters["problem_type"] == "segmentation":
                if target.shape[1] == 3:
                    target = target[:, 0, ...].unsqueeze(1)
                    # this warning should only come up once
                    if accuracy_parameters["print_rgb_label_warning"]:
                        print(
                            "WARNING: The label image is an RGB image, only the first channel will be used.",
                            flush=True,
                        )
                        accuracy_parameters["print_rgb_label_warning"] = False

                if accuracy_parameters["model"]["dimension"] == 2:
                    target = torch.squeeze(target, -1)

        if accuracy_parameters["model"]["dimension"] == 2:
            if "value_keys" in accuracy_parameters:
                if target is not None:
                    if len(target.shape) > 1:
                        target = torch.squeeze(target, -1)

        attention_map = None

        if "medcam_enabled" in accuracy_parameters and accuracy_parameters["medcam_enabled"]:
            output, attention_map = output

        final_loss, final_metric = get_loss_and_metrics(
                image, target, output, accuracy_parameters
        )
        
        if "calib_metrics" in accuracy_parameters:
            metric = accuracy_parameters["calib_metrics"]
        else:
            metric = accuracy_parameters["metrics"][0]
        
        if total_calib_metric == 0:
            total_calib_metric = np.array(final_metric[metric])
        else:
            total_calib_metric += np.array(final_metric[metric])

    average_calib_metric = total_calib_metric / count
     
    return average_calib_metric

def PtqManager(
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

    model, ptq_dataloader = generate_calib_data(
        calib_data=dataframe_calib,
        output_dir=outputDir,
        params=parameters,
    )

    calibration_dataset = nncf.Dataset(ptq_dataloader, transform_fn)
    validation_dataset = nncf.Dataset(ptq_dataloader, transform_fn)
    
    getMetricsParams(parameters)

    if os.path.exists(model_dir):
        core = ov.Core()
        model = core.read_model(model_dir)
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
    
    print("Current PTQ type is: ", parameters['ptq_type'] )
    if parameters['ptq_type'] == 'Default':
        quantized_model = nncf.quantize(
            model,
            calibration_dataset,
            target_device=nncf.TargetDevice.CPU,
        )
    elif parameters['ptq_type'] == 'AccuracyAware':
        quantized_model = nncf.quantize_with_accuracy_control(model,
                        calibration_dataset=calibration_dataset,
                        validation_dataset=validation_dataset,
                        validation_fn=validate,
                        max_drop=0.01)
    else:
        sys.exit("ERROR: 'ptq_type' config parameter is invalid. Valid options: Default, AccuracyAware")
    
    ov.serialize(quantized_model, os.path.join(outputDir, os.path.basename(model_dir)))
    
