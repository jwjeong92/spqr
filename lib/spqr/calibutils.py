import torch
import torch.nn as nn
from .modelutils import get_layers, find_sublayers, get_sequential_groups
from .spqr_engine import SPQRUtil
from .modelutils import FALCON_TYPES 
from .datautils import get_loaders
import time
from tqdm import trange

@torch.no_grad()
def get_inps(model, data_iterable, dev, nsamples=None):
    """mocks model launch to collect inputs to the first model layer"""
    print("catching inputs from data", flush=True)

    layers = get_layers(model)

    if isinstance(data_iterable, torch.Tensor):

        def batch_generator(testenc, seqlen, nsamples):
            for i in range(nsamples):
                batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(dev)
                yield batch

        data_iterable = batch_generator(data_iterable, model.seqlen, nsamples)

    emb = model.get_input_embeddings()
    emb_dev = emb.weight.device
    if emb_dev.type != "cuda":
        emb = emb.to(dev)
        # opt has other embeddings
        if model.config.model_type == "opt":
            model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
            if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
                model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    dev = emb.weight.device  # now default device is the one where the embeddings are.
    layer_dev = next(layers[0].parameters()).device
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)

    forward_arg_names = [
        "attention_mask",
    ]
    if model.config.model_type.lower() in FALCON_TYPES:
        forward_arg_names.append("alibi")

    cache = {"i": 0, "attention_mask": None, "alibi": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            for forward_arg_name in forward_arg_names:
                cache[forward_arg_name] = kwargs.get(forward_arg_name)
            raise ValueError

    layers[0] = Catcher(layers[0])
    saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(min(16, saved_num_threads))
    for batch in data_iterable:
        try:
            if isinstance(batch, (list, tuple)):
                model(batch[0].to(dev))
            elif isinstance(batch, torch.Tensor):
                model(batch.to(dev))
        except ValueError:
            pass
    torch.set_num_threads(saved_num_threads)
    layers[0] = layers[0].module

    layers[0] = layers[0].to(layer_dev)
    model.get_input_embeddings().to(emb_dev)
    if model.config.model_type == "opt":
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(emb_dev)
        if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(emb_dev)
    torch.cuda.empty_cache()

    forward_args = {k: cache[k] for k in forward_arg_names}
    return inps, forward_args
@torch.no_grad()
def insert_catcher(model, dataloader, args, device):
    print("\nStarting sensitivity collection ...")

    inps, forward_args = get_inps(
        model,
        dataloader,
        dev=device,
        nsamples=args.nsamples,
    )
    outs = torch.zeros_like(inps)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = get_layers(model)

    perm_dead_ordered_H_inv_diag = {}
    for i in range(len(layers)):
        perm_dead_ordered_H_inv_diag[i] = {}
        print(f"\n---------------- Layer {i} of {len(layers)} ----------------")
        start_time = time.time()
        
        layer_dev_original = next(layers[i].parameters()).device
        print(f'{layer_dev_original=}')
        if layer_dev_original.type != "cuda":
            layer = layers[i].to(device)
        else:
            layer = layers[i]
        layer_dev = next(layers[i].parameters()).device
        all_sublayers = find_sublayers(layer)

        for k, v in forward_args.items():
            forward_args[k] = v.to(layer_dev) if isinstance(v, torch.Tensor) else v
        
        seqeuntial = [list(all_sublayers.keys())]
        
        for names in seqeuntial:
            subset = {n: all_sublayers[n] for n in names}

            spqr_handler = {}
            for sublayer_name in subset:
                spqr_handler[sublayer_name] = SPQRUtil(
                    subset[sublayer_name]
                )
            def add_batch(name):
                def tmp(_, inp, out):
                    spqr_handler[name].add_batch(inp[0].data)
                
                return tmp
            
            handles = []
            for sublayer_name in subset:
                handles.append(subset[sublayer_name].register_forward_hook(
                    add_batch(sublayer_name)
                ))
            for j in trange(args.nsamples, desc="calc outs before quantization",leave=False):
                outs[j] = layer(inps[j].to(layer_dev).unsqueeze(0), **forward_args)[0]
            for h in handles:
                h.remove()
            
            torch.cuda.empty_cache()
            for sublayer_name in subset:
                print(f"Collecting quant_bf-ed of module {sublayer_name} of layer {i}")
                perm_dead_ordered_H_inv_diag[i][sublayer_name] = spqr_handler[sublayer_name].collect_H_inv(
                    percdamp=args.percdamp,
                    permutation_order=args.permutation_order,
                )

    return perm_dead_ordered_H_inv_diag

def get_hessian_matrix(model, args, device):
    tick = time.time()
    print("Loading data ...")
    dataloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model_path=args.model_path,
        seqlen=model.seqlen,
    )

    results = insert_catcher(model, dataloader, args, device)

    print(f"Calibration time: {time.time() - tick:.1f}")
    return results