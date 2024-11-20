from __future__ import annotations

import math
from typing import NamedTuple, Optional, Union, Dict

import torch
from tqdm.auto import tqdm
import gc
import numpy as np

from .quant_groups import Quantizer, dequantize, quantize
from .weight_permutation import get_permutation_order

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


def inject_bit_errors_packed(X, packed_error_mask):
    # 원본 데이터와 패킹된 오류 마스크에 XOR 연산 적용
    X_int = X.to(torch.int32)  # 원본 데이터의 정수형 변환
    modified_X_int = X_int ^ packed_error_mask  # XOR 연산으로 오류 주입
    return modified_X_int.to(X.dtype)  # 원래 데이터 타입으로 변환

def verify_error_injection(final_error_mask, error_injection_mask, bitwidth=8):
    # 오류 주입이 정상적으로 이루어졌는지 확인
    
    # 1. 오류 주입 갯수 확인
    injected_errors = final_error_mask.sum().item()
    print(f"총 오류 주입 비트 수: {injected_errors}")
    
    # 2. 오류가 outlier 제외한 위치에만 주입되었는지 확인
    eligible_positions = error_injection_mask.sum().item()
    mask_eligible_errors = final_error_mask[error_injection_mask].sum().item()
    print(f"outlier 제외 위치에서 발생한 오류 수: {mask_eligible_errors}")
    
    if injected_errors == mask_eligible_errors:
        print("검증 성공: 오류가 정상적으로 outlier가 아닌 위치에만 주입되었습니다.")
    else:
        print("검증 실패: 일부 오류가 outlier 위치에 주입되었습니다.")


class SPQRUtil:
    """Learns GPTQ for a single linear layer"""

    def __init__(self, layer):
        self.layer = layer
        self.dev = layer.weight.device
        self.columns = self.layer.weight.data.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp):
        assert self.H is not None, "Already ran quantization; cannot add more data batches"
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]

        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()

        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def collect_H_inv(
        self,
        *,
        percdamp: float = 1e-2,
        permutation_order: Union[str, torch.Tensor] = "identity",
        keep_H: bool = True,
    ):
        '''
        weight = self.layer.weight.detach().to(dtype=torch.float, copy=True)
        perm = get_permutation_order(self.H, weight, permutation_order)
        
        weight = weight[:, perm]  # note: weight is modified
        H = self.H
        if keep_H:
            H = H.clone()  # protect from in-place changes
        else:
            self.H = None

        H = H[perm][:, perm]
        self.dead = torch.diag(H) == 0  # indices of input features that do not affect outputs
        if percdamp > 0:
            ix = torch.arange(len(H), device=weight.device)
            H[ix, ix] += percdamp * abs(torch.diag(H)).mean()
            del ix
        H[self.dead, self.dead] = 1
        weight[:, self.dead] = 0
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        H_inv_diag = torch.diag(H_inv)

        if permutation_order != "identity":
            invperm = torch.argsort(perm)
            H_inv_diag = H_inv_diag[invperm]
        '''

        return self.H.to("cpu")

    def collect_quant_loss(
        self,
        *,
        bits: int = 4,
        percdamp: float = 1e-2,
        groupsize: Optional[int] = None,
        keep_last_columns: int = 0,
        permutation_order: Union[str, torch.Tensor] = "identity",
        keep_H: bool = True,
        perchannel: bool = True,
        sym: bool = False,
        verbose=True,
        eval_ppl=True,
        **kwargs,
    ):
        weight = self.layer.weight.detach().to(dtype=torch.float, copy=True)
        perm = get_permutation_order(self.H, weight, permutation_order)
        
        weight = weight[:, perm]  # note: weight is modified
        H = self.H
        if keep_H:
            H = H.clone()  # protect from in-place changes
        else:
            self.H = None

        H = H[perm][:, perm]
        self.dead = torch.diag(H) == 0  # indices of input features that do not affect outputs
        if percdamp > 0:
            ix = torch.arange(len(H), device=weight.device)
            H[ix, ix] += percdamp * abs(torch.diag(H)).mean()
            del ix
        H[self.dead, self.dead] = 1
        weight[:, self.dead] = 0
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        H_inv_diag = torch.diag(H_inv)
        del H, H_inv

        out_dim, in_dim = weight.shape  # [out_features, in_features]

        if groupsize is None:
            groupsize = in_dim

        # groupwise quantization
        assert in_dim % groupsize == 0

        group_start_iter = range(0, in_dim - keep_last_columns, groupsize)
        group_start_iter = tqdm(group_start_iter, leave=False) if verbose else group_start_iter

        dequantized_weight = torch.zeros_like(weight)
        quant_sensitivity = torch.zeros_like(weight)
        for group_start in group_start_iter:
            group_weight = weight[:, group_start : group_start + groupsize]

            group_diag_hessian_inv = H_inv_diag[group_start : group_start + groupsize]

            quantizer = Quantizer()
            quantizer.configure(bits, perchannel=perchannel, sym=sym, **kwargs)
            quantizer.find_params(x=group_weight, weight=True) # get scales, zeros for the weight tensor
            
            group_reconstructed_weight = quantizer.quantize_dequantize(group_weight)

            group_weight_sensitivity = (
                ((group_reconstructed_weight - group_weight).square() / group_diag_hessian_inv)
            )
            quant_sensitivity[:, group_start : group_start + groupsize] = group_weight_sensitivity
            dequantized_weight[:, group_start : group_start + groupsize] = group_reconstructed_weight

        if permutation_order != "identity":
            invperm = torch.argsort(perm)
            quant_sensitivity = quant_sensitivity[:, invperm]
            dequantized_weight = dequantized_weight[:, invperm]
            
        if eval_ppl:
            self.layer.weight.data = dequantized_weight.to(self.layer.weight.dtype)
        return quant_sensitivity

    def collect_quant_bf_loss(
        self,
        layer_name,
        *,
        bits: int = 3,
        percdamp: float = 1e-2,
        groupsize: Optional[int] = None,
        keep_last_columns: int = 0,
        permutation_order: Union[str, torch.Tensor] = "identity",
        keep_H: bool = True,
        perchannel: bool = True,
        sym: bool = False,
        verbose=True,
        ber: float = 1e-4,
        seed: int = 42,
        percentile: int = 100,
        error_extract=False,
        with_sign=False,
        target_layer=None,
        **kwargs,
    ):
        weight = self.layer.weight.detach().to(dtype=torch.float, copy=True)
        perm = get_permutation_order(self.H, weight, permutation_order)
        
        weight = weight[:, perm]  # note: weight is modified
        H = self.H
        if keep_H:
            H = H.clone()  # protect from in-place changes
        else:
            self.H = None

        H = H[perm][:, perm]
        self.dead = torch.diag(H) == 0  # indices of input features that do not affect outputs
        if percdamp > 0:
            ix = torch.arange(len(H), device=weight.device)
            H[ix, ix] += percdamp * abs(torch.diag(H)).mean()
            del ix
        H[self.dead, self.dead] = 1
        weight[:, self.dead] = 0
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        H_inv_diag = torch.diag(H_inv)
        del H, H_inv

        out_dim, in_dim = weight.shape  # [out_features, in_features]

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

        group_start_iter = range(0, in_dim - keep_last_columns, groupsize)
        group_start_iter = tqdm(group_start_iter, leave=False) if verbose else group_start_iter

        for group_start in group_start_iter:
            group_end = min(group_start + groupsize, in_dim)
            in_group_index += 1
            group_weight = weight[:, group_start : group_start + groupsize]

            group_diag_hessian_inv = H_inv_diag[group_start : group_start + groupsize]

            quantizer = Quantizer()
            quantizer.configure(bits, perchannel=perchannel, sym=sym, **kwargs)
            quantizer.find_params(x=group_weight, weight=True) # get scales, zeros for the weight tensor
            
            quant_stats["scales"].append(quantizer.scale)
            quant_stats["zeros"].append(quantizer.zero)

            group_reconstructed_weight = quantize(group_weight, quantizer.scale, quantizer.zero, quantizer.maxq)
            quant_weight[:, group_start : group_start + groupsize] = group_reconstructed_weight
            group_reconstructed_weight = dequantize(group_reconstructed_weight, quantizer.scale, quantizer.zero)

            if with_sign:
                group_weight_sensitivity = (
                    (group_reconstructed_weight - group_weight) / group_diag_hessian_inv
                )
            else:
                group_weight_sensitivity = (
                    ((group_reconstructed_weight - group_weight).square() / group_diag_hessian_inv)
                )
            quant_sensitivity[:, group_start : group_start + groupsize] = group_weight_sensitivity

        if layer_name in target_layer:
            # generate error matrix
            bf_tensor = calculate_bit_error_injection_mask_quantized(
                quant_sensitivity,
                ber = ber,
                seed = seed,
                bitwidth = bits,
                percentile=percentile # wo outlier
            ).reshape_as(quant_weight).to(quant_weight.device)
            if error_extract:
                if permutation_order != "identity":
                    invperm = torch.argsort(perm)
                    bf_tensor = bf_tensor[:, invperm]
                return bf_tensor.to(dtype=torch.int8, device="cpu")
        else:
            bf_tensor = torch.zeros_like(quant_weight)
        
        quant_weight = quant_weight.to(torch.int32) ^ bf_tensor.to(dtype=torch.int32)
        bf_dequant_weight = torch.zeros_like(weight)
        for i in range(len(quant_stats['scales'])):
            bf_dequant_weight[:, i*groupsize : (i+1)*groupsize] = dequantize(
                quant_weight[:, i*groupsize : (i+1)*groupsize], 
                quant_stats['scales'][i], 
                quant_stats['zeros'][i]
            )
        if with_sign:
            bf_sensitivity = (
                ((bf_dequant_weight - weight) / H_inv_diag)
            )
        else:
            bf_sensitivity = (
                ((bf_dequant_weight - weight).square() / H_inv_diag)
            )

        if permutation_order != "identity":
            invperm = torch.argsort(perm)
            bf_dequant_weight = bf_dequant_weight[:, invperm]
            bf_sensitivity = bf_sensitivity[:, invperm]

        self.layer.weight.data = bf_dequant_weight.to(self.layer.weight.dtype)

        return bf_sensitivity.to(dtype=torch.half, device="cpu")

    def quant_mask_error(
        self,
        *,
        bits: int = 3,
        percdamp: float = 1e-2,
        permutation_order: Union[str, torch.Tensor] = "identity",
        keep_H: bool = True,
        perchannel: bool = True,
        sym: bool = False,
        ber: float = 1e-4,
        seed: int = 42,
        percentile: float = 100,
        error_pattern = None,
        **kwargs,
    ):
        weight = self.layer.weight.detach().to(dtype=torch.float, copy=True)
        perm = get_permutation_order(self.H, weight, permutation_order)
        
        weight = weight[:, perm]  # note: weight is modified
        H = self.H
        if keep_H:
            H = H.clone()  # protect from in-place changes
        else:
            self.H = None

        H = H[perm][:, perm]
        self.dead = torch.diag(H) == 0  # indices of input features that do not affect outputs
        if percdamp > 0:
            ix = torch.arange(len(H), device=weight.device)
            H[ix, ix] += percdamp * abs(torch.diag(H)).mean()
            del ix
        H[self.dead, self.dead] = 1
        weight[:, self.dead] = 0
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        H_inv_diag = torch.diag(H_inv)
        del H, H_inv

        out_dim, in_dim = weight.shape  # [out_features, in_features]

        quantizer = Quantizer(weight.shape)
        quantizer.configure(bits=bits, perchannel=perchannel, sym=sym)
        quantizer.find_params(weight, weight=True)

        scale_order = torch.argsort(quantizer.scale.T.squeeze(), descending=True)
        inv_scale_order = torch.argsort(scale_order)

        qweight = quantize(weight, quantizer.scale, quantizer.zero, quantizer.maxq)
        qweight = qweight[scale_order, :]
        err_matrix = calculate_bit_error_injection_mask_quantized(
            qweight, ber, seed, bits, percentile=100
        ).reshape(qweight.shape)
        err_mask_row = round(weight.shape[0] * (percentile/100))
        err_mask_col = round(weight.shape[1] * (percentile/100))
        if error_pattern == 'pattern1':
            err_matrix[ : err_mask_row, err_mask_col : ] = 0
            err_matrix[ : , : err_mask_col] = 0
        elif error_pattern == 'pattern2':
            err_matrix[ : err_mask_row, : err_mask_col] = 0
            err_matrix[err_mask_row : , :] = 0
        elif error_pattern == 'pattern3':
            err_matrix[ : , err_mask_col : ] = 0
            err_matrix[ : err_mask_row, : err_mask_col] = 0
        elif error_pattern == 'pattern2_3':
            err_matrix[ : err_mask_row, : err_mask_col] = 0
            err_matrix[err_mask_row : , err_mask_col : ] = 0
        else:
            err_matrix[:err_mask_row, :err_mask_col] = 0
            
        qweight = qweight.to(torch.int32) ^ err_matrix.to(qweight.device)
        qweight = qweight[inv_scale_order, :]
        dqweight = dequantize(qweight, quantizer.scale, quantizer.zero)
        
        invperm = torch.argsort(perm)
        dqweight = dqweight[:, invperm]
        self.layer.weight.data = dqweight.to(self.layer.weight.dtype)

    def quantize(
        self,
        *,
        bits: int = 2,
        blocksize: int = 128,
        percdamp: float = 1e-2,
        groupsize: Optional[int] = None,
        keep_last_columns: int = 0,
        outlier_relative_threshold: float = float("inf"),
        permutation_order: Union[str, torch.Tensor] = "identity",
        keep_H: bool = True,
        simplified_outliers: bool = False,
        verbose=True,
        perchannel: bool = True,
        sym: bool = False,
        save_quantization: bool = False,
        **kwargs,
    ) -> QuantizationResult:
        """
        :param bits: number of bits used at the lowest level (the full model size will be different!)
        :param blocksize: take blocks of this many input features at a time for GPTQ
        :note: blocksize affects runtime and memory, but does not affect the resulting matrix (up to machine precision)
        :param groupsize: fit quantization scaling / statistics to each group of this many input features
        :param percdamp: relative regularizer added to hessian diagonal before inversion
        :note: if groupsize_in_dim* is None, use the same quantization statistics across all input features
        :param keep_last_columns: if not None, keep the last (this many) input features un_quantized and return them
        :note: the un-quantized columns will be a part of the first returned result
        :param outlier_relative_threshold: threshold used for *UNSTRUCTURED* outliers, relative to
        :note: if keep_last_columns > 0, quantized_dequantized_weights[-keep_last_columns:] will be non-quantized
        :param permutation_order: re-order input features using a certain policy
        :param keep_H: if False, delete the accumulated hessian during quantize; if False, keep the accumulated hessian
        :param simplified_outliers: if True,do not perform leave-one-out evaluation when detecting outliers;
            works faster, but generally worse in perplexity
        :param verbose: if True, display a tqdm progressbar over input columns
        :param sym: if True, base weight quantization is symmetric
        :param perchannel: if True, base weight quantization will learn statistics for each output dimension separately
        :return: a QuantizationResult tuple that contains(
            weight, perm, _unused, _unused, _unused, _unused, quantization_errors, outlier_unstructured_mask
        ), see class QuantizationResult below for details
        """
        weight = self.layer.weight.detach().to(dtype=torch.float, copy=True)
        save_quant_dict = {}
        perm = get_permutation_order(self.H, weight, permutation_order)

        if save_quantization:
            save_quant_dict["quant_weights"] = []
            save_quant_dict["quant_layer_scale"] = []
            save_quant_dict["quant_layer_zeros"] = []
            save_quant_dict["quant_layer_scale_qq_scale"] = []
            save_quant_dict["quant_layer_scale_qq_zero"] = []
            save_quant_dict["quant_layer_zero_qq_scale"] = []
            save_quant_dict["quant_layer_zero_qq_zero"] = []
            save_quant_dict["save_float_dtype"] = self.layer.weight.dtype
            save_quant_dict["outliers_matrix"] = torch.zeros(
                weight.shape, dtype=save_quant_dict["save_float_dtype"]
            ).to(
                weight.device
            )  # shape = [out_features, in_features]

        weight = weight[:, perm]  # note: weight is modified
        H = self.H
        if keep_H:
            H = H.clone()  # protect from in-place changes
        else:
            self.H = None

        H = H[perm][:, perm]
        self.dead = torch.diag(H) == 0  # indices of input features that do not affect outputs
        if percdamp > 0:
            ix = torch.arange(len(H), device=weight.device)
            H[ix, ix] += percdamp * abs(torch.diag(H)).mean()
            del ix
        H[self.dead, self.dead] = 1
        weight[:, self.dead] = 0
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        H_inv_cho = torch.linalg.cholesky(H_inv, upper=True)
        H_inv_cho_diag = torch.diag(H_inv_cho)
        del H

        quantizer = Quantizer()
        quantizer.configure(bits, perchannel=perchannel, sym=sym, **kwargs)
        assert H_inv_cho.shape[0] == H_inv_cho.shape[1] == weight.shape[1], "weight must be [out_features, in_features]"
        out_dim, in_dim = weight.shape  # [out_features, in_features]

        if groupsize is None:
            groupsize = in_dim

        # prepare outlier detection
        outlier_column_indices = torch.empty(0, dtype=torch.int64, device=weight.device)
        del H_inv

        outlier_scale = (weight.var(dim=0) / torch.diag(H_inv_cho).square()).mean().item()
        unstructured_outlier_threshold = outlier_relative_threshold * outlier_scale
        in_group_index = -1  # index of current group of input features, for group quantizer purposes

        quantization_errors = torch.zeros_like(weight)
        unstructured_outlier_mask = torch.zeros_like(weight, dtype=torch.bool)

        block_start_iter = range(0, in_dim - keep_last_columns, blocksize)
        block_start_iter = tqdm(block_start_iter, leave=False) if verbose else block_start_iter
        for block_start in block_start_iter:
            block_end = min(block_start + blocksize, in_dim)
            for column_index in range(block_start, block_end):
                if column_index % groupsize == 0:
                    # fit weight quantizer on the upcoming group of weight columns (inputs), across all rows (outputs)
                    in_group_index += 1
                    group_weight = weight[:, column_index : column_index + groupsize]

                    if simplified_outliers or (unstructured_outlier_threshold == float("inf")):
                        quantizer.find_params(group_weight, weight=True)

                    else:
                        # objective: detect which weights will be designated as outliers, fit quantizer *without* these weights
                        # step 1: fit quantizer on a leave-one-out version of weights, i.e. in each group, drop one weight at a time
                        assert perchannel, "refitting quantizer is only implemented for perchannel=True"
                        group_diag_hessian_inv_cho = H_inv_cho_diag[column_index : column_index + groupsize]
                        loo_quantization_error_sq = get_leave_one_out_error(
                            group_weight, group_diag_hessian_inv_cho, bits=bits, sym=sym
                        )
                        # ^-- dequantized(quantized(group_weight)) using a quantizer trained on all weights except the reconstructed one

                        likely_unstructured_outlier_mask = (
                            loo_quantization_error_sq > unstructured_outlier_threshold
                        ).float()

                        non_outlier_mask = 1 - likely_unstructured_outlier_mask
                        mean_over_non_outliers = torch.sum(
                            group_weight * non_outlier_mask, dim=1, keepdim=True
                        ) / torch.sum(non_outlier_mask, dim=1, keepdim=True).clamp_min(1)
                        group_weight_without_outliers = group_weight * non_outlier_mask + mean_over_non_outliers * (
                            1 - non_outlier_mask
                        )
                        quantizer.find_params(group_weight_without_outliers, weight=True)
                        del group_diag_hessian_inv_cho, loo_quantization_error_sq
                        del mean_over_non_outliers, group_weight_without_outliers, non_outlier_mask

                    if save_quantization:
                        if quantizer.qq_scale_bits is not None:
                            save_quant_dict["quant_layer_scale"].append(quantizer.quant_scale.to(torch.int8))
                            save_quant_dict["quant_layer_scale_qq_scale"].append(
                                quantizer.qq_scale.scale.to(save_quant_dict["save_float_dtype"])
                            )
                            save_quant_dict["quant_layer_scale_qq_zero"].append(
                                quantizer.qq_scale.zero.to(save_quant_dict["save_float_dtype"])
                            )
                        else:
                            save_quant_dict["quant_layer_scale"].append(
                                quantizer.scale.to(save_quant_dict["save_float_dtype"])
                            )

                        if quantizer.qq_zero_bits is not None and (
                            (not quantizer.round_zero) or quantizer.qq_zero_bits < quantizer.bits
                        ):
                            save_quant_dict["quant_layer_zeros"].append(quantizer.quant_zero.to(torch.int8))
                            save_quant_dict["quant_layer_zero_qq_scale"].append(
                                quantizer.qq_zero.scale.to(save_quant_dict["save_float_dtype"])
                            )
                            save_quant_dict["quant_layer_zero_qq_zero"].append(
                                quantizer.qq_zero.zero.to(save_quant_dict["save_float_dtype"])
                            )
                        else:
                            save_quant_dict["quant_layer_zeros"].append(
                                quantizer.zero.to(save_quant_dict["save_float_dtype"])
                            )
                    del group_weight

                weight_quant_i = quantize(
                    weight[:, column_index].unsqueeze(1), quantizer.scale, quantizer.zero, quantizer.maxq
                )
                weight_i_quantized = dequantize(weight_quant_i, quantizer.scale, quantizer.zero).reshape_as(
                    weight[:, column_index]
                )

                delta_weight_i = weight[:, column_index] - weight_i_quantized  # [out_dim]
                quantization_errors[:, column_index] = (
                    delta_weight_i / H_inv_cho[column_index, column_index]
                )  # [out_dim]

                if unstructured_outlier_threshold != float("inf"):
                    unstructured_outlier_mask[:, column_index] = (
                        quantization_errors[:, column_index].square() > unstructured_outlier_threshold
                    )
                    # re-quantize without outliers
                    is_outlier = unstructured_outlier_mask[:, column_index].float()

                    weight_quant_i = quantize(
                        (weight[:, column_index] * (1 - is_outlier)).unsqueeze(1),
                        quantizer.scale,
                        quantizer.zero,
                        quantizer.maxq,
                    )
                    weight_i_quantized_wo_outliers = dequantize(
                        weight_quant_i, quantizer.scale, quantizer.zero
                    ).reshape_as(weight[:, column_index])
                    weight_i_quantized = (
                        weight_i_quantized_wo_outliers * (1 - is_outlier) + weight[:, column_index] * is_outlier
                    )  # [out_dim]

                    if save_quantization:
                        save_quant_dict["outliers_matrix"][:, column_index] = weight[:, column_index] * is_outlier

                    del weight_i_quantized_wo_outliers

                    delta_weight_i = weight[:, column_index] - weight_i_quantized  # [out_dim]
                    quantization_errors[:, column_index] = (
                        delta_weight_i / H_inv_cho[column_index, column_index]
                    )  # [out_dim]

                if save_quantization:
                    save_quant_dict["quant_weights"].append(weight_quant_i.to(torch.int8))

                weight[:, column_index] = weight_i_quantized
                weight[:, column_index + 1 : block_end].addr_(
                    quantization_errors[:, column_index],
                    H_inv_cho[column_index, column_index + 1 : block_end],
                    alpha=-1,
                )

            weight[:, block_end:].addmm_(
                quantization_errors[:, block_start:block_end],
                H_inv_cho[block_start:block_end, block_end:],
                alpha=-1,
            )

        if permutation_order != "identity":
            invperm = torch.argsort(perm)
            weight = weight[:, invperm]

        if save_quantization:
            save_quant_dict["perm"] = perm.to(torch.int32)
            save_quant_dict["keep_last_columns"] = 0
            save_quant_dict["blocksize"] = 128
            save_quant_dict["weight_shape"] = weight.shape
            save_quant_dict["groupsize"] = groupsize if groupsize else weight.shape[1]
            save_quant_dict["quant_weights"] = torch.cat(save_quant_dict["quant_weights"], dim=1)
            save_quant_dict["outliers_matrix"] = save_quant_dict["outliers_matrix"].to_sparse()

        return QuantizationResult(
            weight=weight,
            perm=perm,
            quantization_errors=quantization_errors,
            unstructured_outlier_threshold=unstructured_outlier_threshold,
            unstructured_outlier_mask=unstructured_outlier_mask,
            save_quant_dict=save_quant_dict,
        )


class QuantizationResult(NamedTuple):
    """A collection of codebooks, indices and assorted statistics produced by SPQRUtil; not memory-optimized!"""

    weight: torch.FloatTensor  # dequantized(quantized(weight)), same shape as the original
    perm: Optional[torch.LongTensor]  # optional input permutation indices that were used during quantization
    # NOTE: if permutation_order != identity, all subsequent tensors (incl. outlier indices) are permuted in that order!

    quantization_errors: torch.Tensor  # per-element quantization errors, defined as (weight - quantized_weight) / diag(inverse_hessian_cholesky)
    unstructured_outlier_threshold: float  # threshold on squared error increase used for determining *UNSTRUCTURED* outliers
    unstructured_outlier_mask: torch.Tensor  # bool mask where True means that this is an individual outlier
    save_quant_dict: dict

def get_sensitivity_parallel(p_factor, weight, H_inv_diag, *, bits, sym):
    sensitivity = torch.zeros_like(weight, device=weight.device)

    # as a baseline error, quantize data normally without outliers
    base_quantizer = Quantizer(shape=weight.shape)
    base_quantizer.configure(bits, perchannel=True, sym=sym)
    base_quantizer.find_params(weight, weight=True)
    baseline_reconstructed_weights = base_quantizer.quantize_dequantize(weight)
    baseline_errors_sq = (
        ((baseline_reconstructed_weights - weight) / H_inv_diag).square().sum(dim=1, keepdim=True)
    )

    loo_indices = torch.arange(weight.shape[1], device=weight.device)
    loo_indices = loo_indices[1:] - (loo_indices[:, None] >= loo_indices[1:]).to(loo_indices.dtype)

    for col_idx in range(0, weight.shape[1], p_factor):    
        p_idx = loo_indices[col_idx : col_idx+p_factor]
        groupwise_loo_data = weight[:, p_idx]
        fast_quantizer = Quantizer(shape=groupwise_loo_data.flatten(0, 1).shape)
        fast_quantizer.configure(bits, perchannel=True, sym=sym)
        fast_quantizer.find_params(groupwise_loo_data.flatten(0, 1), weight=True)

        # compute error improvement from not quantizing each one weight
        # to do so, we shall first train quantizer on leave-one-out data (which can be done faster since not all data affects quantization)
        loo_groupwise_reconstructed_weights = fast_quantizer.quantize_dequantize(
            groupwise_loo_data.flatten(0, 1)
        ).reshape_as(groupwise_loo_data)
        loo_group_diag_hessian_inv_cho = H_inv_diag[p_idx]  # [num_loo = groupsize, groupsize - 1]
        assert H_inv_diag.ndim == 1

        # total quantization error consists of hessian-weighted mse on all remaining weights except for the one that's left out
        # -- this is because the left-out weights will not be quantized, and therefore, has zero quantization error
        loo_errors_sq = (
            ((loo_groupwise_reconstructed_weights - groupwise_loo_data) / loo_group_diag_hessian_inv_cho).square().sum(-1)
        )
        assert loo_errors_sq.shape[1] == p_factor  # [num_groups, num_loo = groupsize]


        # outlier's usefulness = how much does mse decrease from treating this weight as an outlier
        sensitivity[:, col_idx:col_idx+p_factor] = baseline_errors_sq - loo_errors_sq
    
    return sensitivity

def get_leave_one_out_error(group_weight: torch.Tensor, group_diag_hessian_inv_cho: torch.Tensor, *, bits, sym):
    """EXPERIMENTAL! BEWARE - for each weight, fit quantizer without this_one_weight and return this one weight's reconstruction"""

    assert group_weight.ndim == 2
    loo_indices = torch.arange(group_weight.shape[1], device=group_weight.device)
    loo_indices = loo_indices[1:] - (loo_indices[:, None] >= loo_indices[1:]).to(loo_indices.dtype)
    groupwise_loo_data = group_weight[:, loo_indices]  # [num_groups, num_loo = groupsize, groupsize - 1]
    fast_quantizer = Quantizer(shape=groupwise_loo_data.flatten(0, 1).shape)
    fast_quantizer.configure(bits, perchannel=True, sym=sym)
    fast_quantizer.find_params(groupwise_loo_data.flatten(0, 1), weight=True)

    # compute error improvement from not quantizing each one weight
    # to do so, we shall first train quantizer on leave-one-out data (which can be done faster since not all data affects quantization)
    loo_groupwise_reconstructed_weights = fast_quantizer.quantize_dequantize(
        groupwise_loo_data.flatten(0, 1)
    ).reshape_as(groupwise_loo_data)
    loo_group_diag_hessian_inv_cho = group_diag_hessian_inv_cho[loo_indices]  # [num_loo = groupsize, groupsize - 1]
    assert group_diag_hessian_inv_cho.ndim == 1

    # total quantization error consists of hessian-weighted mse on all remaining weights except for the one that's left out
    # -- this is because the left-out weights will not be quantized, and therefore, has zero quantization error
    loo_errors_sq = (
        ((loo_groupwise_reconstructed_weights - groupwise_loo_data) / loo_group_diag_hessian_inv_cho).square().sum(-1)
    )
    assert loo_errors_sq.shape == group_weight.shape  # [num_groups, num_loo = groupsize]

    # as a baseline error, quantize data normally without outliers
    base_quantizer = Quantizer(shape=group_weight.shape)
    base_quantizer.configure(bits, perchannel=True, sym=sym)
    base_quantizer.find_params(group_weight, weight=True)
    baseline_reconstructed_weights = base_quantizer.quantize_dequantize(group_weight)
    baseline_errors_sq = (
        ((baseline_reconstructed_weights - group_weight) / group_diag_hessian_inv_cho).square().sum(dim=1, keepdim=True)
    )

    # outlier's usefulness = how much does mse decrease from treating this weight as an outlier
    reduction_in_squared_error = baseline_errors_sq - loo_errors_sq
    return reduction_in_squared_error
