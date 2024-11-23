import scipy.sparse as sp
import torch
import numpy as np

def conv_bin2int(bin, wbits):
    bin = bin.reshape(wbits, -1)
    sum = torch.zeros(bin.size(-1), dtype=torch.int32)
    for i in range(len(bin)):
        sum += bin[i] * 2**((wbits-1)-i)
    return sum
    
def error_gen(param, rate, seed, wbits):
    orig_size = param.size()
    bitwidth = param.data.element_size()*8
    
    bin_error = torch.tensor(sp.random(np.prod(orig_size), wbits, density=rate, dtype=bool, random_state=np.random.default_rng(seed)).toarray())
    error_matrix = conv_bin2int(bin_error, wbits)
    del bin_error
    return error_matrix.view(orig_size)

def error_injection(param, rate, seed, wbits, device="cuda"):
    err_mat = error_gen(param, rate, seed, wbits).to(device)
    int_form = err_mat.dtype
    if param.element_size() == 2:
        return err_mat.to(torch.int16)
    elif param.element_size() == 1:
        return err_mat.to(torch.int8)
    else:
        return err_mat.to(torch.int32)