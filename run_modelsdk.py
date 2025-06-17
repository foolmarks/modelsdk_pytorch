'''
**************************************************************************
||                        SiMa.ai CONFIDENTIAL                          ||
||   Unpublished Copyright (c) 2022-2023 SiMa.ai, All Rights Reserved.  ||
**************************************************************************
 NOTICE:  All information contained herein is, and remains the property of
 SiMa.ai. The intellectual and technical concepts contained herein are 
 proprietary to SiMa and may be covered by U.S. and Foreign Patents, 
 patents in process, and are protected by trade secret or copyright law.

 Dissemination of this information or reproduction of this material is 
 strictly forbidden unless prior written permission is obtained from 
 SiMa.ai.  Access to the source code contained herein is hereby forbidden
 to anyone except current SiMa.ai employees, managers or contractors who 
 have executed Confidentiality and Non-disclosure agreements explicitly 
 covering such access.

 The copyright notice above does not evidence any actual or intended 
 publication or disclosure  of this source code, which includes information
 that is confidential and/or proprietary, and is a trade secret, of SiMa.ai.

 ANY REPRODUCTION, MODIFICATION, DISTRIBUTION, PUBLIC PERFORMANCE, OR PUBLIC
 DISPLAY OF OR THROUGH USE OF THIS SOURCE CODE WITHOUT THE EXPRESS WRITTEN
 CONSENT OF SiMa.ai IS STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE 
 LAWS AND INTERNATIONAL TREATIES. THE RECEIPT OR POSSESSION OF THIS SOURCE
 CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS TO 
 REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE, USE, OR
 SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.                

**************************************************************************
'''

"""
Quantize and compile the PyTorch model.
Usage from inside Palette docker: python run_modelsdk.py
"""

"""
Author: Mark Harvey
"""

import os
import sys
import argparse
import numpy as np
import logging
import tarfile

# Palette-specific imports
from afe.load.importers.general_importer import ImporterParams, pytorch_source
from afe.apis.defines import default_quantization
from afe.apis.loaded_net import load_model
from afe.apis.error_handling_variables import enable_verbose_error_messages
from afe.apis.release_v1 import get_model_sdk_version
from afe.core.utils import length_hinted

# user imports
import config as cfg
DIVIDER = cfg.DIVIDER

# pre-processing
def _preprocessing(image):
    """
    Normalize, Mean subtraction, div by std deviation
    Add batch dimension
    """
    image = cfg.preprocess(image)
    image = image[np.newaxis, :, :, :]
    return np.float32(image)


def implement(args):
    logger = logging.getLogger(__name__)
    enable_verbose_error_messages()

    # get filename from full path
    filename = os.path.splitext(os.path.basename(args.model_path))[0]

    # set an output path for saving results
    output_path = f'{args.build_dir}/{filename}'

    # load the floating-point model
    input_names = ['x']
    input_shapes = [(1, 3, 224, 224)]
    importer_params: ImporterParams = pytorch_source(args.model_path, input_names, input_shapes)
    loaded_net = load_model(importer_params)

    # calibration data
    with np.load(args.calib_data) as data:
        calib_images = data['x']
        logger.info("Number of calibration images: %d", calib_images.shape[0])

    calibration_data = []
    for img in calib_images:
        preproc_image = _preprocessing(img)
        calibration_data.append({input_names[0]: preproc_image})

    # quantize
    quant_model = loaded_net.quantize(
        calibration_data=length_hinted(len(calib_images), calibration_data),
        quantization_config=default_quantization,
        model_name=filename,
        log_level=logging.WARN
    )
    quant_model.save(model_name=filename, output_directory=output_path)
    logger.info("Quantized and saved to %s", output_path)

    # evaluate quantized model
    logger.info("Evaluating quantized model...")
    with np.load(args.test_data) as data:
        test_images = data['x']
        labels = data['y']
        logger.info("Number of test images: %d", test_images.shape[0])

    correct = 0
    for i, img in enumerate(test_images):
        img = _preprocessing(img)
        test_data = {input_names[0]: img}
        prediction = quant_model.execute(test_data, fast_mode=True)
        prediction = np.argmax(prediction)
        if prediction == labels[i]:
            correct += 1
    accuracy = correct / len(labels) * 100
    logger.info("Correct predictions: %d Accuracy: %.2f%%", correct, accuracy)

    # compile
    quant_model.compile(
        output_path=output_path,
        batch_size=args.batch_size,
        log_level=logging.INFO)
    logger.info("Compiled model written to %s", output_path)

    # extract compiled model
    model_tar = f'{output_path}/{filename}_mpk.tar.gz'
    with tarfile.open(model_tar) as model:
        model.extractall(output_path)


def run_main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-bd', '--build_dir', type=str, default='build', help='Path of build folder. Default is build')
    ap.add_argument('-m', '--model_path', type=str, default='./pyt/resnext101_32x8d_wsl.pt', help='path to FP32 model')
    ap.add_argument('-b', '--batch_size', type=int, default=1, help='requested batch size of compiled model. Default is 1')
    ap.add_argument('-td', '--test_data', type=str, default='test_data.npz', help='Path of test data numpy file. Default is test_data.npz')
    ap.add_argument('-cd', '--calib_data', type=str, default='calib_data.npz', help='Path of calibration data numpy file. Default is calib_data.npz')
    args = ap.parse_args()

    # ensure build directory exists
    os.makedirs(args.build_dir, exist_ok=True)

    # configure logging to 'run_modelsdk.log'
    log_file_path = 'run_modelsdk.log'
    logging.basicConfig(
        level=logging.INFO,
        filename=log_file_path,
        filemode='w',
        format='%(asctime)s %(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info(DIVIDER)
    logger.info("Model SDK version %s", get_model_sdk_version())
    logger.info(sys.version)
    logger.info(DIVIDER)

    implement(args)


if __name__ == '__main__':
    run_main()
