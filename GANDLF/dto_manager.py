import torch
import nncf.torch  # Important - must be imported before any other external package that depends on torch

import nncf
from nncf import NNCFConfig
from nncf.torch import create_compressed_model, load_state, register_default_init_args
import os, sys

from openvino.runtime import CompiledModel
import numpy as np
from GANDLF.compute.loss_and_metric import get_loss_and_metrics
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from GANDLF.logger import Logger
from GANDLF.metrics import overall_stats
from GANDLF.compute.training_loop import train_network
from GANDLF.compute.forward_pass import validate_network
import time
from tqdm import tqdm
import torchio

from GANDLF.optimizers.nncf_optimizer import get_parameter_groups, make_optimizer
from GANDLF.utils import (
    get_date_time,
    best_model_path_end,
    latest_model_path_end,
    initial_model_path_end,
    save_model,
    optimize_and_save_model,
    load_model,
    version_check,
    write_training_patches,
    print_model_summary,
    get_ground_truths_and_predictions_tensor,
    get_model_dict,
    print_and_format_metrics,
)

from GANDLF.compute  import create_pytorch_objects

from nncf.torch.initialization import PTInitializingDataLoader

from torch.utils.data import TensorDataset, DataLoader

def InitDataLoader(train_dataloader, optimizer, params):
    images = None
    labels = None

    for batch_idx, (subject) in enumerate(
        tqdm(train_dataloader, desc="Looping over training data")
    ):
        optimizer.zero_grad()
        image = (
            torch.cat(
                [subject[key][torchio.DATA] for key in params["channel_keys"]], dim=1
            )
            .float()
            .to(params["device"])
        )
        if images == None:
            images = torch.squeeze(image)
        else:
            images = torch.cat((images, torch.squeeze(image)), dim = 0)

        # images.append(image.cpu().numpy())
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
        if labels == None:
            labels = torch.squeeze(label)
        else:
            labels = torch.cat((labels, torch.squeeze(label)), dim = 0)
    
    tensor_x = torch.squeeze(torch.Tensor(images)) # transform to torch tensor
    tensor_y = torch.squeeze(torch.Tensor(labels))

    init_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    init_dataloader = DataLoader(init_dataset, batch_size = params["batch_size"]) # create your dataloader


    return(init_dataloader)


def get_default_weight_decay(config):
    compression_configs = config.get("compression", {})
    if isinstance(compression_configs, dict):
        compression_configs = [compression_configs]

    weight_decay = 1e-4
    for compression_config in compression_configs:
        if compression_config.get("algorithm") == "rb_sparsity":
            weight_decay = 0

    return weight_decay

def getMetricsParams(parameters):
    global accuracy_parameters
    accuracy_parameters = parameters

def DtoManager(
    dataframe_train,
    dataframe_validation,
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

    # Load a configuration file to specify compression
    nncf_config = NNCFConfig.from_dict(parameters['nncf_dict'])

    if "batch_size" in nncf_config:
        parameters["batch_size"] = nncf_config["batch_size"]

    model_paths = {
        "best": os.path.join(
            outputDir, parameters["model"]["architecture"] + best_model_path_end
        ),
        "initial": os.path.join(
            outputDir, parameters["model"]["architecture"] + initial_model_path_end
        ),
        "latest": os.path.join(
            outputDir, parameters["model"]["architecture"] + latest_model_path_end
        ),
    }

    # if previous model file is present, load it up for sanity checks
    main_dict = None
    if os.path.exists(model_paths["best"]):
        main_dict = load_model(model_paths["best"], parameters["device"].lower(), False)
        # version_check(parameters["version"], version_to_check=main_dict["version"])
        # parameters["previous_parameters"] = main_dict.get("parameters", None)

    # Defining our model here according to parameters mentioned in the configuration file
    parameters['problem_type'] = 'classification'
    print("Number of channels : ", parameters["model"]["num_channels"])

    (
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        scheduler,
        parameters,
    ) = create_pytorch_objects(parameters, dataframe_train, dataframe_validation, parameters["device"].lower())

    #train_dataloader = init_dto_dataloader()


    model.load_state_dict(main_dict["model_state_dict"]) 
    optimizer.load_state_dict(main_dict["optimizer_state_dict"])

    init_dataloader = InitDataLoader(train_dataloader, optimizer, parameters)
    # now you pass this wrapped object instead of your original dataloader into the `register_default_init_args`
    nncf_config = register_default_init_args(nncf_config, init_dataloader)
    # and then call `create_compressed_model` with that config file as usual.

    compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)

    # datetime object containing current date and time
    print("Initializing training at :", get_date_time(), flush=True)

    calculate_overall_metrics = (parameters["problem_type"] == "classification") or (
        parameters["problem_type"] == "regression"
    )

    # get the overall metrics that are calculated automatically for classification/regression problems
    if parameters["problem_type"] == "regression":
        overall_metrics = overall_stats(torch.Tensor([1]), torch.Tensor([1]), parameters)
    elif parameters["problem_type"] == "classification":
        # this is just used to generate the headers for the overall stats
        temp_tensor = torch.randint(0, parameters["model"]["num_classes"], (5,))
        overall_metrics = overall_stats(
            temp_tensor.to(dtype=torch.int32),
            temp_tensor.to(dtype=torch.int32),
            parameters,
        )

    metrics_log = parameters["metrics"].copy()
    if calculate_overall_metrics:
        for metric in overall_metrics:
            if metric not in metrics_log:
                metrics_log[metric] = 0


    nncf_train_logger = Logger(
        logger_csv_filename=os.path.join(outputDir, "nncf_logs_training.csv"),
        metrics=metrics_log,
    )
    nncf_valid_logger = Logger(
        logger_csv_filename=os.path.join(outputDir, "nncf_logs_validation.csv"),
        metrics=metrics_log,
    )

    compression_scheduler = compression_ctrl.scheduler
    # switch to train mode
    compressed_model.train()

    # define optimizer
    params_to_optimize = get_parameter_groups(compressed_model, nncf_config)
    nncf_optimizer, nncf_lr_scheduler = make_optimizer(params_to_optimize, nncf_config)

    if 'epochs' in nncf_config:
        epochs = nncf_config['epochs']
    else:
        epochs = 10


    for epoch in range(epochs):
        compression_scheduler.epoch_step()

        # Printing times
        epoch_start_time = time.time()
        print("*" * 20)
        print("*" * 20)
        print("Starting Epoch : ", epoch)
        if parameters["verbose"]:
            print("Epoch start time : ", get_date_time())

        parameters["current_epoch"] = epoch


        epoch_train_loss, epoch_train_metric = train_network(
            compressed_model, train_dataloader, nncf_optimizer, parameters
        )
        compression_loss = compression_ctrl.loss()
        loss = epoch_train_loss + compression_loss
        epoch_valid_loss, epoch_valid_metric = validate_network(
            compressed_model, val_dataloader, compression_scheduler, parameters, epoch, mode="validation"
        )


        # Write the losses to a logger
        nncf_train_logger.write(epoch, epoch_train_loss, epoch_train_metric)
        nncf_valid_logger.write(epoch, epoch_valid_loss, epoch_valid_metric)

        



