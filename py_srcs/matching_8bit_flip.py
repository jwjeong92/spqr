import torch
import numpy as np

model_name = 'opt-125m'
base_path = '/raid/jwjeong/quant_results'
wbits = 6
groupsize = None
BERs = [
    '1e-04',
    '2e-04',
    '3e-04',
    '4e-04',
    '5e-04',
    '6e-04',
    '7e-04',
    '8e-04',
    '9e-04',
    '1e-03',
]
seed = 42

if groupsize is None:
    sensitivity_for_base = torch.load(f'{base_path}/{model_name}-w{wbits}.pt')
    quant_bf_results_path = '/raid/jwjeong/quant_bf_results_nogroups'
else:
    sensitivity_for_base = torch.load(f'{base_path}/{model_name}-w{wbits}-gs{groupsize}.pt')
    quant_bf_results_path = f'/raid/jwjeong/quant_bf_results_gs{groupsize}'

value_sens = {}
fc1_sens = {}
for ber in BERs:
    quant_bf_results = torch.load(f'{quant_bf_results_path}/{model_name}-w{wbits}-bf{ber}-seed{seed}.pt')
    for layernum, sublayer in quant_bf_results.items():
        if layernum not in fc1_sens:
            fc1_sens[layernum] = []
            value_sens[layernum] = []
        for name, sensitivity in sublayer.items():
            if 'fc1' in name:
                fc1_sens[layernum].append(sensitivity.sum(dim=1))
            if 'v_proj' in name:
                value_sens[layernum].append(sensitivity.sum(dim=1))

base_model = torch.load(f'{base_path}/opt-125m-w6.pt')