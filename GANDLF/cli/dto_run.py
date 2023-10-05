from pathlib import Path

import pandas as pd
import os, sys, pickle, subprocess, shutil
from sklearn.model_selection import KFold

from GANDLF.ptq_manager import PtqManager
from GANDLF.dto_manager import DtoManager
from GANDLF.parseConfig import parseConfig
from GANDLF.utils import (
    populate_header_in_parameters,
    parseTrainingCSV,
)

import numpy as np


def dto_run(
    data_csv, config_file, model_dir, output_dir=None
):
    """
    Main function that runs the training and inference.

    Args:
        data_csv (str): The CSV files of the training and validation data for during training optimization, separated by comma.
        config_file (str): The YAML file of the training configuration.
        model_dir (str): The model directory for the trained model.
        output_dir (str): The output directory for the NNCF ptq session.

    Returns:
        None
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)
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
    if "," in data_csv:
        # training and validation pre-split
        data_full = None
        all_csvs = data_csv.split(",")
        data_train, headers_train = parseTrainingCSV(all_csvs[0], train=True)
        data_validation, headers_validation = parseTrainingCSV(
            all_csvs[1], train=True
        )
        parameters = populate_header_in_parameters(parameters, headers_train)
        assert (
            headers_train == headers_validation
        ), "The training and validation CSVs do not have the same header information."
    else:

        kf_validation = KFold(n_splits=parameters["nested_training"]["validation"])

        data_train, headers_train = parseTrainingCSV(data_csv, train=True)
        parameters = populate_header_in_parameters(parameters, headers_train)
        # split across subjects
        subjectIDs_full = (
            data_train[data_train.columns[parameters["headers"]["subjectIDHeader"]]]
            .unique()
            .tolist()
        )

        # get the indeces for kfold splitting
        trainingData_full = data_train

        # start the kFold train for validation
        for train_index, val_index in kf_validation.split(
            subjectIDs_full
        ):
            trainingData = pd.DataFrame()  # initialize the variable
            validationData = pd.DataFrame()  # initialize the variable

            # loop over all train_index and construct new dataframe
            for subject_idx in train_index:
                trainingData = pd.concat([trainingData, 
                    trainingData_full[
                        trainingData_full[
                            trainingData_full.columns[
                                parameters["headers"]["subjectIDHeader"]
                            ]
                        ]
                        == subjectIDs_full[subject_idx]
                    ]]
                )

            # loop over all val_index and construct new dataframe
            for subject_idx in val_index:
                validationData = pd.concat([validationData,
                    trainingData_full[
                        trainingData_full[
                            trainingData_full.columns[
                                parameters["headers"]["subjectIDHeader"]
                            ]
                        ]
                        == subjectIDs_full[subject_idx]
                    ]]
                )

    DtoManager(
                dataframe_train=trainingData,
                dataframe_validation=validationData,
                model_dir = model_dir, 
                outputDir=parameters["output_dir"],
                parameters=parameters,
    )
