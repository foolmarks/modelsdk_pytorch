

import torch


#model_list = ['resnext101_32x8d_wsl','resnext101_32x16d_wsl','resnext101_32x32d_wsl', 'resnext101_32x48d_wsl']
model_list = ['resnext101_32x8d_wsl']

# input format is NCHW
input_shape = [1, 3, 224, 224]

for model_name in model_list:
    model = torch.hub.load('facebookresearch/WSL-Images', model_name)
    model.eval()
    # save as scripted model
    model_scripted = torch.jit.trace(model, torch.randn(input_shape))
    model_scripted.save(model_name+'.pt')
    print('saved',model_name)
    
