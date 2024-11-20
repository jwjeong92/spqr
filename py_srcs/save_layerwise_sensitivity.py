import torch
import numpy as np
import matplotlib.pyplot as plt

def extract_ppl_from_log(log_file_path):
    # 결과를 저장할 리스트
    extracted_lines = []

    # 파일을 열고 라인을 하나씩 읽어 조건에 맞는 라인 추출
    with open(log_file_path, 'r') as file:
        for line in file:
            if line.startswith("eval. quant bf model:"):
                extracted_lines.append(line.strip())  # line 끝의 공백 제거

    ppl_dict = {}

    # 추출한 라인 출력
    for extracted_line in extracted_lines:
        ppl = float(extracted_line.split(' ')[-1])
        ppl_dict
    return extracted_lines

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
seed = 43

if groupsize is None:
    sensitivity_for_base = torch.load(f'{base_path}/{model_name}-w{wbits}.pt')
    quant_bf_results_path = 'quant_bf_results_nogroups_w_sign'
else:
    sensitivity_for_base = torch.load(f'{base_path}/{model_name}-w{wbits}-gs{groupsize}.pt')
    quant_bf_results_path = f'/raid/jwjeong/quant_bf_results_gs{groupsize}'

fc1_sens = {}
for ber in BERs:
    quant_bf_results = torch.load(f'{quant_bf_results_path}/{model_name}-w{wbits}-bf{ber}-seed{seed}.pt')
    for layernum, sublayer in quant_bf_results.items():
        if layernum not in fc1_sens:
            fc1_sens[layernum] = []
        for name, sensitivity in sublayer.items():
            if 'fc1' in name:
                fc1_sens[layernum].append(sensitivity.sum(dim=1))

base_model = torch.load(f'{base_path}/opt-125m-w6.pt')

for j in fc1_sens:
    for i in range(len(fc1_sens[j])):
        baseline = base_model[j]['fc1'].sum(dim=1)
        print((fc1_sens[j][i]).mean().item())

fig, axs = plt.subplots(3, 1, sharex=True)
axs[0].plot(fc1_sens[0][-1], label = '1e-3')
axs[0].plot(fc1_sens[0][1], label = '2e-4')
axs[0].plot(fc1_sens[0][0], label = '1e-4')
axs[1].plot(baseline, label='baseline')
plt.legend()
plt.savefig('temp.png')

ppls = [28.93310546875, 29.775320053100586, 31.966800689697266, 35.62201690673828, 44.51797866821289, 49.51250076293945, 53.6937141418457, 55.85478591918945, 63.50621795654297, 66.04556274414062]
