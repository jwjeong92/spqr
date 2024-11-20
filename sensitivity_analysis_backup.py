# %%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy

local_path = '/raid/LLM/opt-125m/'
model = AutoModelForCausalLM.from_pretrained(local_path, torch_dtype=torch.float)
tokenizer= AutoTokenizer.from_pretrained(local_path)


# %%
from spqr.quant_groups import Quantizer, quantize, dequantize, quantize_dequantize
import gc

cp_model = deepcopy(model)
cp_model = cp_model.to("cuda")
layers = cp_model.model.decoder.layers
linear_weights = {}
num_layer = len(layers)
for i in range(num_layer):
    linear_weights[i] = {
        'self_attn.k_proj': layers[i].self_attn.k_proj.weight.detach().to(dtype=torch.float, copy=True),
        'self_attn.q_proj': layers[i].self_attn.q_proj.weight.detach().to(dtype=torch.float, copy=True),
        'self_attn.v_proj': layers[i].self_attn.v_proj.weight.detach().to(dtype=torch.float, copy=True),
        'self_attn.out_proj': layers[i].self_attn.out_proj.weight.detach().to(dtype=torch.float, copy=True),
        'fc1': layers[i].fc1.weight.detach().to(dtype=torch.float, copy=True),
        'fc2': layers[i].fc2.weight.detach().to(dtype=torch.float, copy=True),
    }
H_dict = torch.load('collected_H/H_opt-125m_pajama_seed43.pt')

def calculate_bit_error_injection_mask_quantized(X, ber=1e-2, seed=42, bitwidth=8, percentile=99):
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    # 상위 1%를 outlier로 지정하는 mask 생성
    threshold = torch.quantile(X, percentile / 100.0) if percentile != 100 else torch.max(X)
    outlier_mask = (X > threshold).to(torch.float32)
    
    # 오류를 주입할 대상 위치: outlier가 아닌 부분
    error_injection_mask = (outlier_mask == 0).to(torch.bool)
    
    # 유효 비트 수 기준으로 전체 비트 수 계산 및 BER 목표에 맞는 비트 오류 수 계산
    total_bits = X.numel() * bitwidth
    target_error_bits = int(total_bits * ber)

    # 오류를 주입할 수 있는 eligible 비트 위치
    eligible_indices = torch.nonzero(error_injection_mask.flatten(), as_tuple=False).view(-1)
    eligible_bits = eligible_indices.numel() * bitwidth
    
    if eligible_bits < target_error_bits:
        print("경고: 주어진 BER을 맞추기에 충분한 비트가 없습니다.")
        target_error_bits = eligible_bits

    # eligible_indices에서 비트 위치별 인덱스 생성
    eligible_bit_indices = eligible_indices.repeat_interleave(bitwidth) * bitwidth + torch.arange(bitwidth).repeat(eligible_indices.size(0)).to(eligible_indices.device)

    # 무작위로 target_error_bits 개수만큼 선택
    selected_bit_indices = eligible_bit_indices[torch.randperm(eligible_bit_indices.size(0), generator=gen)[:target_error_bits]]

    # 최종 오류 마스크 생성
    final_error_mask = torch.zeros(X.numel() * bitwidth, dtype=torch.bool)
    final_error_mask[selected_bit_indices] = True

    packed_error_mask = torch.zeros(X.numel(), dtype=torch.int32)
    for i in range(bitwidth):
        packed_error_mask |= (final_error_mask.view(X.numel(), bitwidth)[:, i].int() << i)
    

    # 비트 위치별 패킹을 벡터화하여 수행
    #shifts = 2 ** torch.arange(bitwidth, dtype=torch.int32)  # [1, 2, 4, 8, ..., 2^(bitwidth-1)]
    #packed_error_mask = (final_error_mask.view(X.numel(), bitwidth).int() * shifts).sum(dim=1)

    return packed_error_mask

def quant_and_masked_error_injection(linear_weights, H_dict, wbits, ber, seed):
    start_seed = seed
    percdamp = 1e0
    err_dqweight = {}
    for i in linear_weights:
        err_dqweight[i] = {}
        for name in linear_weights[i]:
            weight = linear_weights[i][name]
            H = H_dict[i][name].to("cuda")
            act_perm = torch.argsort(torch.diag(H), descending=True)
            act_invperm = torch.argsort(act_perm)
            
            weight = weight[:, act_perm]
            H = H[act_perm][:, act_perm]
            dead = torch.diag(H) == 0
            if percdamp > 0:
                ix = torch.arange(len(H), device=weight.device)
                H[ix, ix] += percdamp * abs(torch.diag(H)).mean()
                del ix
            H[dead, dead] = 1
            weight[:, dead] = 0
            H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            H_inv_diag = torch.diag(H_inv)
            del H, H_inv

            out_dim, in_dim = weight.shape

            quantizer = Quantizer(weight.shape)
            quantizer.configure(wbits, True, False)
            quantizer.find_params(weight, weight=True)

            scale_order = torch.argsort(quantizer.scale.T.squeeze(), descending=True)
            inv_scale_order = torch.argsort(scale_order)
            ordered_scale = quantizer.scale[scale_order, :]
            ordered_zero = quantizer.scale[scale_order, :]

            weight = weight[scale_order, :]

            qweight = quantize(weight, ordered_scale, ordered_zero, quantizer.maxq)
            err_matrix = calculate_bit_error_injection_mask_quantized(
                weight, ber, start_seed, wbits, percentile=100
            ).reshape_as(weight)
            err_mask_row = round(weight.shape[0] * 0.01)
            err_mask_col = round(weight.shape[1] * 0.01)
            err_mask = torch.ones(weight.shape, dtype=bool)
            err_mask[:err_mask_row, :err_mask_col] = False
            err_matrix = err_matrix * err_mask
            qweight = qweight.to(dtype=torch.int32, device="cuda") ^ err_matrix.to(dtype=torch.int32, device="cuda")
            dqweight = dequantize(qweight, ordered_scale, ordered_zero)
            dqweight = dqweight[:, act_invperm]
            dqweight = dqweight[inv_scale_order, :]
            err_dqweight[i][name] = dqweight.to(weight.dtype)
            start_seed += 10
    return err_dqweight

wbits = 6
ber = 0
seed = 43

err_dqweight = quant_and_masked_error_injection(linear_weights, H_dict, wbits, ber, seed)
for i in err_dqweight:
    for name in err_dqweight[i]:
        if 'k_proj' in name:
            layers[i].self_attn.k_proj.weight.data = err_dqweight[i][name]
        elif 'q_proj' in name:
            layers[i].self_attn.q_proj.weight.data = err_dqweight[i][name]
        elif 'v_proj' in name:
            layers[i].self_attn.v_proj.weight.data = err_dqweight[i][name]
        elif 'out_proj' in name:
            layers[i].self_attn.out_proj.weight.data = err_dqweight[i][name]
        elif 'fc1' in name:
            layers[i].fc1.weight.data = err_dqweight[i][name]
        elif 'fc2' in name:
            layers[i].fc2.weight.data = err_dqweight[i][name]

from datasets import load_dataset
from tqdm import tqdm
import torch.nn as nn
def evaluate_perplexity(model, tokenizer):
    def _perplexity(nlls, n_samples, seqlen):
        return torch.exp(torch.stack(nlls).sum() / (n_samples * seqlen))

    # load and prepare dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    data = tokenizer("\n\n".join(data["text"]), return_tensors="pt")
    data = data.input_ids.to(model.device)

    seqlen = 2048
    model = model.eval()
    n_samples = data.numel() // seqlen

    nlls = []

    with tqdm(range(n_samples), desc="Perplexity -") as progress_bar:
        for i in progress_bar:
            start_index = i * seqlen
            end_index = (i + 1) * seqlen
            batch = data[:, start_index:end_index].to(model.device)
            with torch.no_grad():
                logits = model(batch).logits
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = data[:, start_index:end_index][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

            curr_ppl = _perplexity(nlls, i + 1, seqlen)
            progress_bar.set_description(f"Perplexity {curr_ppl:.3f}")

    ppl = _perplexity(nlls, n_samples, seqlen)

    return ppl.item()

evaluate_perplexity(cp_model.to("cuda"), tokenizer)


# %%
gc.collect()
torch.cuda.empty_cache()

# %%
# collect basic sensitivity
from spqr.quant_groups import Quantizer, quantize, dequantize, quantize_dequantize
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter

percdamp = 1e0
wbits = 6
groupsize = 16
start_seed = 43

# return sensitivity, scale, zero, qweight, H_inv_diag_dict
def calc_base_sens(linear_weights, H_dict, percdamp, wbits):
    scale = {}
    zero = {}
    qweight = {}
    orig_weight = {}
    sensitivity = {}
    H_inv_diag_dict = {}
    for i in linear_weights:
        scale[i] = {}
        zero[i] = {}
        qweight[i] = {}
        sensitivity[i] = {}
        orig_weight[i] = {}
        H_inv_diag_dict[i] = {}
        for name in linear_weights[i]:
            weight = linear_weights[i][name]
            H = H_dict[i][name].to("cuda")
            perm = torch.argsort(torch.diag(H), descending=True)
            invperm = torch.argsort(perm)
            
            weight = weight[:, perm]
            H = H[perm][:, perm]
            dead = torch.diag(H) == 0
            if percdamp > 0:
                ix = torch.arange(len(H), device=weight.device)
                H[ix, ix] += percdamp * abs(torch.diag(H)).mean()
                del ix
            H[dead, dead] = 1
            weight[:, dead] = 0
            orig_weight[i][name] = weight[:, invperm]
            H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            H_inv_diag = torch.diag(H_inv)
            H_inv_diag_dict[i][name] = H_inv_diag
            del H, H_inv

            out_dim, in_dim = weight.shape
            quantizer = Quantizer(weight.shape)
            quantizer.configure(bits=wbits, perchannel=True, sym=False)
            quantizer.find_params(x=weight, weight=True)
            scale[i][name] = quantizer.scale
            zero[i][name] = quantizer.zero
            qweight[i][name] = quantize(weight, quantizer.scale, quantizer.zero, quantizer.maxq).to(torch.int8)
            dqweight = quantize_dequantize(weight, quantizer.scale, quantizer.zero, quantizer.maxq)
            sensitivity[i][name] = ((dqweight - weight).square() / H_inv_diag)[:, invperm]
    
    return sensitivity, scale, zero, qweight, H_inv_diag_dict, orig_weight
# return sensitivity, scale, zero, qweight, H_inv_diag_dict, orig_weight
def calc_group_quant_sens(linear_weights, H_dict, percdamp, wbits, groupsize):
    scale = {}
    zero = {}
    qweight = {}
    orig_weight = {}
    sensitivity = {}
    H_inv_diag_dict = {}
    for i in linear_weights:
        scale[i] = {}
        zero[i] = {}
        qweight[i] = {}
        sensitivity[i] = {}
        orig_weight[i] = {}
        H_inv_diag_dict[i] = {}
        for name in linear_weights[i]:
            weight = linear_weights[i][name]
            
            H = H_dict[i][name].to("cuda")
            perm = torch.argsort(torch.diag(H), descending=True)
            invperm = torch.argsort(perm)
            
            weight = weight[:, perm]
            H = H[perm][:, perm]
            dead = torch.diag(H) == 0
            if percdamp > 0:
                ix = torch.arange(len(H), device=weight.device)
                H[ix, ix] += percdamp * abs(torch.diag(H)).mean()
                del ix
            H[dead, dead] = 1
            weight[:, dead] = 0
            orig_weight[i][name] = weight
            H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            H_inv_diag = torch.diag(H_inv)
            H_inv_diag_dict[i][name] = H_inv_diag
            del H, H_inv

            out_dim, in_dim = weight.shape

            if groupsize is None:
                groupsize = in_dim

            # groupwise quantization
            assert in_dim % groupsize == 0
            in_group_index = -1
            
            quant_stats = {
                "scales": [],
                "zeros": [],
            }
            
            quant_weight = torch.zeros_like(weight)
            quant_sensitivity = torch.zeros_like(weight)

            group_start_iter = range(0, in_dim, groupsize)

            for group_start in group_start_iter:
                group_end = min(group_start + groupsize, in_dim)
                in_group_index += 1
                group_weight = weight[:, group_start : group_start + groupsize]

                group_diag_hessian_inv = H_inv_diag[group_start : group_start + groupsize]

                quantizer = Quantizer()
                quantizer.configure(wbits, perchannel=True, sym=False)
                quantizer.find_params(x=group_weight, weight=True) # get scales, zeros for the weight tensor
                
                quant_stats["scales"].append(quantizer.scale)
                quant_stats["zeros"].append(quantizer.zero)

                group_reconstructed_weight = quantize(group_weight, quantizer.scale, quantizer.zero, quantizer.maxq)
                quant_weight[:, group_start : group_start + groupsize] = group_reconstructed_weight
                group_reconstructed_weight = dequantize(group_reconstructed_weight, quantizer.scale, quantizer.zero)

                group_weight_sensitivity = (
                    ((group_reconstructed_weight - group_weight).square() / group_diag_hessian_inv)
                )
                quant_sensitivity[:, group_start : group_start + groupsize] = group_weight_sensitivity

            quant_sensitivity = quant_sensitivity[:, invperm]

            sensitivity[i][name] = quant_sensitivity
            scale[i][name] = quant_stats["scales"]
            zero[i][name] = quant_stats["zeros"]
            qweight[i][name] = quant_weight[:, invperm]
            H_inv_diag_dict[i][name] = H_inv_diag

    return sensitivity, scale, zero, qweight, H_inv_diag_dict, orig_weight
# return sensitivity, scale, zero, qweight, H_inv_diag_dict
def calc_group_quant_err_sens(linear_weights, H_dict, percdamp, wbits, groupsize, ber, seed, percentile):
    scale = {}
    zero = {}
    qweight = {}
    orig_weight = {}
    sensitivity = {}
    bf_map = {}
    H_inv_diag_dict = {}
    start_seed = seed
    for i in linear_weights:
        scale[i] = {}
        zero[i] = {}
        qweight[i] = {}
        sensitivity[i] = {}
        orig_weight[i] = {}
        bf_map[i] = {}
        H_inv_diag_dict[i] = {}
        for name in linear_weights[i]:
            weight = linear_weights[i][name]
            orig_weight[i][name] = weight
            H = H_dict[i][name].to("cuda")
            perm = torch.argsort(torch.diag(H), descending=True)
            invperm = torch.argsort(perm)
            
            weight = weight[:, perm]
            H = H[perm][:, perm]
            dead = torch.diag(H) == 0
            if percdamp > 0:
                ix = torch.arange(len(H), device=weight.device)
                H[ix, ix] += percdamp * abs(torch.diag(H)).mean()
                del ix
            H[dead, dead] = 1
            weight[:, dead] = 0
            H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            H_inv_diag = torch.diag(H_inv)
            H_inv_diag_dict[i][name] = H_inv_diag
            del H, H_inv

            out_dim, in_dim = weight.shape

            if groupsize is None:
                groupsize = in_dim

            # groupwise quantization
            assert in_dim % groupsize == 0
            in_group_index = -1
            
            quant_stats = {
                "scales": [],
                "zeros": [],
            }
            
            quant_weight = torch.zeros_like(weight)
            quant_sensitivity = torch.zeros_like(weight)

            group_start_iter = range(0, in_dim, groupsize)

            for group_start in group_start_iter:
                group_end = min(group_start + groupsize, in_dim)
                in_group_index += 1
                group_weight = weight[:, group_start : group_start + groupsize]

                group_diag_hessian_inv = H_inv_diag[group_start : group_start + groupsize]

                quantizer = Quantizer()
                quantizer.configure(wbits, perchannel=True, sym=False)
                quantizer.find_params(x=group_weight, weight=True) # get scales, zeros for the weight tensor
                
                quant_stats["scales"].append(quantizer.scale)
                quant_stats["zeros"].append(quantizer.zero)

                group_reconstructed_weight = quantize(group_weight, quantizer.scale, quantizer.zero, quantizer.maxq)
                quant_weight[:, group_start : group_start + groupsize] = group_reconstructed_weight
                group_reconstructed_weight = dequantize(group_reconstructed_weight, quantizer.scale, quantizer.zero)

                group_weight_sensitivity = (
                    ((group_reconstructed_weight - group_weight).square() / group_diag_hessian_inv)
                )
                quant_sensitivity[:, group_start : group_start + groupsize] = group_weight_sensitivity
            
            bf_tensor = calculate_bit_error_injection_mask_quantized(
                quant_sensitivity,
                ber = ber,
                seed = seed,
                bitwidth = wbits,
                percentile=percentile # wo outlier
            ).reshape_as(quant_weight).to(quant_weight.device)
            #bf_map[i][name] = bf_tensor.to(dtype=torch.int8)
            quant_weight = quant_weight.to(torch.int32) ^ bf_tensor.to(dtype=torch.int32)
            bf_dequant_weight = torch.zeros_like(weight)
            for idx in range(len(quant_stats['scales'])):
                bf_dequant_weight[:, idx*groupsize : (idx+1)*groupsize] = dequantize(
                    quant_weight[:, idx*groupsize : (idx+1)*groupsize], 
                    quant_stats['scales'][idx], 
                    quant_stats['zeros'][idx]
                )
            bf_sensitivity = (
                ((bf_dequant_weight - weight).square() / H_inv_diag)
            )
            bf_sensitivity = bf_sensitivity[:, invperm]

            sensitivity[i][name] = bf_sensitivity
            scale[i][name] = quant_stats["scales"]
            zero[i][name] = quant_stats["zeros"]
            qweight[i][name] = quant_weight[:, invperm]
            H_inv_diag_dict[i][name] = H_inv_diag

            seed += 10

    return sensitivity, scale, zero, qweight, H_inv_diag_dict

# return loo_sensitivity, loo_scale, loo_zero
def calc_loo_error_sq(linear_weights, H_dict, percdamp, wbits):
    loo_scale = {}
    loo_zero = {}
    loo_sensitivity = {}
    for i in linear_weights:
        loo_scale[i] = {}
        loo_zero[i] = {}
        loo_sensitivity[i] = {}
        for name in linear_weights[i]:
            weight = linear_weights[i][name]
            H = H_dict[i][name].to("cuda")
            perm = torch.argsort(torch.diag(H), descending=True)
            invperm = torch.argsort(perm)
            
            weight = weight[:, perm]
            H = H[perm][:, perm]
            dead = torch.diag(H) == 0
            if percdamp > 0:
                ix = torch.arange(len(H), device=weight.device)
                H[ix, ix] += percdamp * abs(torch.diag(H)).mean()
                del ix
            H[dead, dead] = 1
            weight[:, dead] = 0
            H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
            H_inv_diag = torch.diag(H_inv)
            del H, H_inv

            out_dim, in_dim = weight.shape
            loo_scale[i][name] = torch.zeros(weight.shape, device=weight.device)
            loo_zero[i][name] = torch.zeros(weight.shape, device=weight.device)
            loo_sensitivity[i][name] = torch.zeros(weight.shape, device=weight.device)

            for col_idx in range(in_dim):
                loo_idx = [i for i in range(in_dim) if i != col_idx]
                loo_weight = weight[:, loo_idx]

                quantizer = Quantizer(loo_weight.shape)
                quantizer.configure(bits=6, perchannel=True, sym=False)
                quantizer.find_params(x=loo_weight, weight=True)
                
                loo_scale[i][name][:, col_idx] = quantizer.scale.view(-1)
                loo_zero[i][name][:, col_idx] = quantizer.zero.view(-1)
            
                loo_dqweight = quantize_dequantize(loo_weight, quantizer.scale, quantizer.zero, quantizer.maxq)
                loo_sensitivity[i][name][:, col_idx] = ((loo_dqweight - loo_weight) / H_inv_diag[loo_idx]).square().sum(dim=1)
            
            loo_scale[i][name] = loo_scale[i][name][:, invperm]
            loo_zero[i][name] = loo_zero[i][name][:, invperm]

            quantizer = Quantizer(weight.shape)
            quantizer.configure(bits=wbits, perchannel=True, sym=False)
            quantizer.find_params(x=weight, weight=True)

            dqweight = quantize_dequantize(weight, quantizer.scale, quantizer.zero, quantizer.maxq)
            baseline_error_sq = ((dqweight - weight) / H_inv_diag).square().sum(dim=1, keepdim=True)
            loo_sensitivity[i][name] = baseline_error_sq - loo_sensitivity[i][name]

            loo_sensitivity[i][name] = loo_sensitivity[i][name][:, invperm]
    
    return loo_sensitivity, loo_scale, loo_zero
# return sens, weight, H_inv_diag
def classify_by_layer_from_dict(layer_name, calc_dict):
    temp = []    
    for i in calc_dict:
        for name in calc_dict[i]:
            if layer_name in name:
                temp.append(calc_dict[i][name])
    return temp

def save_2D_sensitivity_map(sens, save_path, filter_size=8):
    for i in range(len(sens)):
        pooled_tensor = maximum_filter(sens[i].to(dtype=torch.float, device="cpu"), size=filter_size)
        plt.figure(figsize=(6, 6))
        plt.imshow(pooled_tensor, cmap='Blues', aspect='auto')
        plt.colorbar(label='Sensitviity')

        plt.xlabel("Input Feature Dimension")
        plt.ylabel("Output Feature Dimension")
        
        plt.savefig(f'{save_path}/fc1_layer{i}', dpi=600)

def plot_weighted_sensitivity(sens):
    for i in range(len(sens)):
        plt.plot(sens[i].sum(dim=1).sqrt().cpu())

        plt.xlabel("Output Feature Dimension")
        plt.ylabel("Weighted Sensitvity")

        plt.show()

def calculate_bit_error_injection_mask_quantized(X, ber=1e-2, seed=42, bitwidth=8, percentile=99):
    gen = torch.Generator()
    gen.manual_seed(seed)
    
    # 상위 1%를 outlier로 지정하는 mask 생성
    threshold = torch.quantile(X, percentile / 100.0) if percentile != 100 else torch.max(X)
    outlier_mask = (X > threshold).to(torch.float32)
    
    # 오류를 주입할 대상 위치: outlier가 아닌 부분
    error_injection_mask = (outlier_mask == 0).to(torch.bool)
    
    # 유효 비트 수 기준으로 전체 비트 수 계산 및 BER 목표에 맞는 비트 오류 수 계산
    total_bits = X.numel() * bitwidth
    target_error_bits = int(total_bits * ber)

    # 오류를 주입할 수 있는 eligible 비트 위치
    eligible_indices = torch.nonzero(error_injection_mask.flatten(), as_tuple=False).view(-1)
    eligible_bits = eligible_indices.numel() * bitwidth
    
    if eligible_bits < target_error_bits:
        print("경고: 주어진 BER을 맞추기에 충분한 비트가 없습니다.")
        target_error_bits = eligible_bits

    # eligible_indices에서 비트 위치별 인덱스 생성
    eligible_bit_indices = eligible_indices.repeat_interleave(bitwidth) * bitwidth + torch.arange(bitwidth).repeat(eligible_indices.size(0)).to(eligible_indices.device)

    # 무작위로 target_error_bits 개수만큼 선택
    selected_bit_indices = eligible_bit_indices[torch.randperm(eligible_bit_indices.size(0), generator=gen)[:target_error_bits]]

    # 최종 오류 마스크 생성
    final_error_mask = torch.zeros(X.numel() * bitwidth, dtype=torch.bool)
    final_error_mask[selected_bit_indices] = True

    packed_error_mask = torch.zeros(X.numel(), dtype=torch.int32)
    for i in range(bitwidth):
        packed_error_mask |= (final_error_mask.view(X.numel(), bitwidth)[:, i].int() << i)
    

    # 비트 위치별 패킹을 벡터화하여 수행
    #shifts = 2 ** torch.arange(bitwidth, dtype=torch.int32)  # [1, 2, 4, 8, ..., 2^(bitwidth-1)]
    #packed_error_mask = (final_error_mask.view(X.numel(), bitwidth).int() * shifts).sum(dim=1)

    return packed_error_mask

def get_seed_list_for_target_layers(layer_name, num_layer, start_seed):
    if 'k_proj' in layer_name:
        seeds_list = [start_seed + 60*x+0 for x in range(len(num_layer))]
    elif 'v_proj' in layer_name:
        seeds_list = [start_seed + 60*x+10 for x in range(len(num_layer))]
    elif 'q_proj' in layer_name:
        seeds_list = [start_seed + 60*x+20 for x in range(len(num_layer))]
    elif 'out_proj' in layer_name:
        seeds_list = [start_seed + 60*x+30 for x in range(len(num_layer))]
    elif 'fc1' in layer_name:
        seeds_list = [start_seed + 60*x+40 for x in range(len(num_layer))]
    elif 'fc2' in layer_name:
        seeds_list = [start_seed + 60*x+50 for x in range(len(num_layer))]
    return seeds_list

def get_error_tensor_for_target_layers(weight, H, layer_name, num_layer, ber, start_seed, wbits, percentile):
    seeds_list = get_seed_list_for_target_layers(layer_name, num_layer, start_seed)
    err_tensor_list = []
    for i, seed in enumerate(seeds_list):
        perm = torch.argsort(torch.diag(H[i][layer_name]), descending=True)
        invperm = torch.argsort(perm)
        err_tensor = calculate_bit_error_injection_mask_quantized(weight, ber, seed, wbits, percentile)
        err_tensor = err_tensor.reshape(weight.shape)[:, invperm].to(weight.device)
        err_tensor_list.append(err_tensor)
    return err_tensor_list


# %%
sensitivity, scale, zero, qweight, H_inv_diag_dict, orig_weight = calc_base_sens(linear_weights, H_dict, percdamp, wbits)
fc1_sens =classify_by_layer_from_dict('fc1', sensitivity)
fc1_scale = classify_by_layer_from_dict('fc1', scale)
fc1_H_inv_diag = classify_by_layer_from_dict('fc1', H_inv_diag_dict)
fc1_orig_weight = classify_by_layer_from_dict('fc1', orig_weight)

# %%
act_perm = torch.argsort(fc1_H_inv_diag[0], descending=True)
actorder_weight = fc1_orig_weight[0][:, act_perm]
actorder_sens = fc1_sens[0][:, act_perm]

# %%
plt.plot(fc1_H_inv_diag[0][act_perm].to("cpu"))

# %%
scale_perm = torch.argsort(fc1_scale[0].T.squeeze(), descending=True)
ordered_weight = actorder_weight[scale_perm, :]
ordered_sens = actorder_sens[scale_perm, :]

# %%
quantizer = Quantizer(fc1_orig_weight[0].shape)
quantizer.configure(wbits, True, False)
quantizer.find_params(fc1_orig_weight[0], weight=True)
original_scale = quantizer.scale
actorder_scale = fc1_scale[0]

# %%
prot_row = round(ordered_weight.shape[0]*0.01)
prot_col = round(ordered_weight.shape[1]*0.01)
prot_weight = ordered_weight[:32, :32].clone()

plt.figure(figsize=(6, 6))
plt.imshow(prot_weight.to("cpu"), cmap='Blues', aspect='auto')
plt.colorbar(label='Sensitviity')

plt.xlabel("Input Feature Dimension")
plt.ylabel("Output Feature Dimension")

# %%


# %%
threshold = torch.quantile(scale[0]['fc1'].view(-1).sort().values.to(torch.float), 0.99)
high_range_rows = (scale[0]['fc1'].view(-1) > threshold).nonzero().view(-1)
for i in range(len(orig_weight[0]['fc1'][high_range_rows,:])):
    plt.plot(orig_weight[0]['fc1'][high_range_rows,:][i].cpu())

# %%
plt.plot(H_inv_diag_dict[0]['fc1'].cpu())

# %%


# %%
ber = 2e-4
seed = 43
percentile = 100
sensitivity, scale, zero, qweight, H_inv_diag_dict = calc_group_quant_err_sens(
    linear_weights, H_dict, percdamp, wbits, groupsize, ber, seed, percentile
    )


# %%
layer_name = 'fc1'
layer_sens = classify_by_layer_from_dict(layer_name, sensitivity)
save_path = 'figures/q6_err_gs16'
save_2D_sensitivity_map(layer_sens, save_path)

# %%
#sensitivity, scale, zero, qweight, H_inv_diag_dict = collect_base_sens(linear_weights, H_dict, percdamp, wbits)
sensitivity, scale, zero, qweight, H_inv_diag_dict, orig_weight = calc_group_quant_sens(linear_weights, H_dict, percdamp, wbits, groupsize)
layer_name = 'fc1'
layer_sens = classify_by_layer_from_dict(layer_name, sensitivity)
layer_weight = classify_by_layer_from_dict(layer_name, orig_weight)
layer_scale = classify_by_layer_from_dict(layer_name, scale)
save_path = 'figures/q6_err'
save_2D_sensitivity_map(layer_sens, save_path)

# %%
for i in range(len(layer_scale[0])):
    print(layer_scale[0][i].view(-1).sort().values)


# %%
top10_sens = fc1_sens[0].sum(dim=1).sort(descending=True).indices[:20]
top10_scales = scale[0]['fc1'].view(-1).sort(descending=True).indices[:20]
print(top10_sens)
print(top10_scales)

# %%
# top-k sensitive weight vs. cross-points sensitive weights
import numpy as np
high_sensitive_rowidx = []
high_sensitive_colidx = []

for i in range(len(fc1_sens)):
    print(f'layer {i}')
    high_row = fc1_sens[i].sum(dim=1).sort(descending=True).indices[:10]
    high_col = fc1_sens[i].sum(dim=0).sort(descending=True).indices[:10]

    cross_points = fc1_sens[i][np.ix_(high_row, high_col)].flatten()
    topk_values, topk_indices = torch.topk(fc1_sens[i].flatten(), 100)

    common_elements = torch.tensor([x for x in cross_points if x in topk_values])
    print(f'{common_elements.numel()=}')
    sorted_indices1 = torch.argsort(cross_points, descending=True)
    ranks1 = torch.zeros(cross_points.shape, dtype=torch.int64)

    ranks1[sorted_indices1] = torch.arange(len(cross_points)) + 1

    # tensor2에서 공통 요소의 순위 찾기
    sorted_indices2 = torch.argsort(topk_values, descending=True)
    ranks2 = torch.zeros(topk_values.shape, dtype=torch.int64)
    ranks2[sorted_indices2] = torch.arange(len(topk_values)) + 1

    # 각 텐서에서 공통 요소의 순위 추출
    common_ranks1 = [ranks1[(cross_points == x).nonzero(as_tuple=True)[0].item()].item() for x in common_elements]
    common_ranks2 = [ranks2[(topk_values == x).nonzero(as_tuple=True)[0].item()].item() for x in common_elements]

    # 결과 출력
    for i, val in enumerate(common_elements):
        print(f"Value {val}: rank in tensor1 = {common_ranks1[i]}, rank in tensor2 = {common_ranks2[i]}")

# %%
# scale factor distributions
scale[0]['fc1'].view(-1).sort()

# %%
# loo scale factor distributions
plt.plot(loo_scale[0]['fc1'].min(dim=1, keepdim=True).indices.view(-1).sort().values.cpu())


