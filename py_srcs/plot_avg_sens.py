import numpy as np
import torch

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import os

ber_list = np.linspace(1, 10, 10) * 1e-4
formatted_ber = [format(x, ".0e") for x in ber_list]
seed_list = list(range(42, 52))
wbits=[4, 6, 8]

def get_out_sum(target):
    for i in target:
        for name in target[i]:
            target[i][name] = target[i][name].sum(1)
    return target

def calculate_average_tensor(X):
    results = {}
    
    # 각 layer_num, layer_name 별로 tensor를 수집하여 평균 계산
    for seed, layers in X.items():
        for layer_num, layer_data in layers.items():
            for layer_name, tensor in layer_data.items():
                # 결과 딕셔너리 초기화
                if layer_num not in results:
                    results[layer_num] = {}
                if layer_name not in results[layer_num]:
                    results[layer_num][layer_name] = []
                
                # 동일한 layer_num, layer_name에 해당하는 tensor를 리스트에 추가
                results[layer_num][layer_name].append(tensor)
    
    # 모든 seed에 대한 평균 계산
    for layer_num, layer_data in results.items():
        for layer_name, tensor_list in layer_data.items():
            # 텐서 리스트를 스택하여 평균 계산
            results[layer_num][layer_name] = torch.mean(torch.stack(tensor_list), dim=0)
    
    return results
    
avg_sens = {}
for ber in formatted_ber:
    avg_sens[ber] = {}
    for wbit in wbits:
        avg_sens[ber][wbit] = {}
        for seed in seed_list:
            file_path = f'quant_bf_results/w{wbit}-results/opt-125m-w{wbit}-bf{ber}-seed{seed}.pt'
            avg_sens[ber][wbit][seed] = get_out_sum(torch.load(file_path))
        
        avg_sens[ber][wbit] = calculate_average_tensor(avg_sens[ber][wbit])

data = []
for BER, wbits in avg_sens.items():
    for wbit, layers in wbits.items():
        for layer_num, layer in layers.items():
            for layer_name, avg_sensitivity in layer.items():
                data.append([BER, wbit, layer_num, layer_name, avg_sensitivity])

df = pd.DataFrame(data, columns=['BER', 'bit-width', 'layer-num', 'layer-name', 'avg_sensitivity'])


def plot_sensitivity_by_bitwidth(df, save_dir="plots_by_bitwidth_468"):
    os.makedirs(save_dir, exist_ok=True)

    # 각 BER, LayerNum, LayerName에 대해 bitwidth 별로 avg_sensitivity를 플롯
    for (BER, layer_num, layer_name), group in df.groupby(['BER', 'layer-num', 'layer-name']):
        plt.figure(figsize=(10, 6))
        
        # 각 Bitwidth에 따른 AvgSensitivity의 평균을 플롯
        for bitwidth, bitwidth_group in group.groupby('bit-width'):
            if bitwidth == 8 or bitwidth == 6 or bitwidth == 4:
                avg_sensitivity = np.mean(np.stack(bitwidth_group['avg_sensitivity'].values), axis=0)
                plt.plot(range(len(avg_sensitivity)), avg_sensitivity, label=f"bit-width {bitwidth}")
        
        # 그래프 레이블과 타이틀 설정
        plt.xlabel("Tensor Index")
        plt.ylabel("Average Sensitivity")
        plt.title(f"BER: {BER}, Layer: {layer_num}-{layer_name} - Sensitivity by Bitwidth")
        plt.legend(title="Bitwidth")
        plt.grid(True)

        # 파일 이름 설정 및 저장
        filename = f"{save_dir}/BER_{BER}_Layer_{layer_num}_{layer_name}_by_bitwidth_468.png"
        plt.savefig(filename)
        plt.close()

# 플롯 생성
plot_sensitivity_by_bitwidth(df)

def plot_sensitivity_by_BER(df, save_dir="plots_by_BER"):
    os.makedirs(save_dir, exist_ok=True)

    # 각 Bitwidth, LayerNum, LayerName에 대해 BER 별로 avg_sensitivity를 플롯
    for (bitwidth, layer_num, layer_name), group in df.groupby(['bit-width', 'layer-num', 'layer-name']):
        plt.figure(figsize=(10, 6))
        
        # 각 BER에 따른 AvgSensitivity의 평균을 플롯
        for BER, BER_group in group.groupby('BER'):
            if BER == '1e-04' or BER == '3e-04' or BER == '9e-04':
                if BER == '1e-04':
                    zorder = 3
                elif BER == '3e-04':
                    zorder = 2
                else:
                    zorder = 1
                avg_sensitivity = np.mean(np.stack(BER_group['avg_sensitivity'].values), axis=0)
                plt.plot(range(len(avg_sensitivity)), avg_sensitivity, label=f"BER {BER}", zorder=zorder)
        
        # 그래프 레이블과 타이틀 설정
        plt.xlabel("Tensor Index")
        plt.ylabel("Average Sensitivity")
        plt.title(f"Bitwidth: {bitwidth}, Layer: {layer_num}-{layer_name} - Sensitivity by BER")
        plt.legend(title="BER")
        plt.grid(True)

        # 파일 이름 설정 및 저장
        filename = f"{save_dir}/Bitwidth_{bitwidth}_Layer_{layer_num}_{layer_name}_by_BER.png"
        plt.savefig(filename)
        plt.close()

# 플롯 생성
plot_sensitivity_by_BER(df)