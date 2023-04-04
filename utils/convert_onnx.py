import numpy as np
from models import *

# from datasets import
import torch
import torch.nn as nn
import torch.distributed as dist


def convert_onnx(model, dummy_input1, dummy_input2):
    dummy_input1.requires_grad = True
    dummy_input2.requires_grad = True
    # dummy_input = torch.tensor(dummy_input, dtype=torch.float, requires_grad=True).cuda()
    # truth = torch.tensor(truth, dtype=torch.float, requires_grad=True).cuda()
    # truth = model(dummy_input,dummy_input)
    model.eval()
    # output = model(dummy_input,dummy_input,truth)
    print('start converting')
    torch.onnx.export(model,         # model being run
                      # model input (or a tuple for multiple inputs)
                      (dummy_input1, dummy_input2),
                      "test.onnx",       # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      verbose=True,
                      opset_version=13,    # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      # the model's input names
                      input_names=['modelInput', 'modelInput2'],
                      # the model's output names
                      output_names=['modelOutput'],
                      #  dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                      #         'modelOutput' : {0 : 'batch_size'}}
                      )
    print(" ")
    print('Model has been converted to ONNX')
