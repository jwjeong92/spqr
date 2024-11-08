import os
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "3"  # Set the GPU 2 to use

import torch
import torch.nn as nn
from tqdm import trange

import pandas as pd

from lib.spqr.datautils import get_loaders
from lib.spqr.modelutils import (
    FALCON_TYPES,
    find_sublayers,
    get_layers,
    get_lm_logits,
    get_model,
    get_model_head,
    get_sequential_groups,
)
from lib.spqr.spqr_engine import Quantizer, SPQRUtil, quantize

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

try:
    import safetensors  # noqa: F401

    has_safetensors = True
except ModuleNotFoundError:
    has_safetensors = False


@torch.no_grad()
def get_inps(model, data_iterable, args, dev, nsamples=None):
    """mocks model launch to collect inputs to the first model layer"""
    print("catching inputs from data", flush=True)

    layers = get_layers(model)

    nsamples = nsamples or args.nsamples

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
def collect_sensitivity(model, dataloader, args, device):
    print("\nStarting sensitivity collection ...")

    inps, forward_args = get_inps(
        model,
        dataloader,
        args,
        dev="cpu" if args.offload_activations else device,
    )
    outs = torch.zeros_like(inps)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    save = getattr(args, "save", False)

    layers = get_layers(model)
    collected_sensitivity = {}
    for i in range(len(layers)):
        collected_sensitivity[i] = {}
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
        
        if args.true_sequential:
            seqeuntial = get_sequential_groups(model)
        else:
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
                if args.offload_activations:
                    outs[j] = outs[j].cpu()
            for h in handles:
                h.remove()
            
            torch.cuda.empty_cache()

            for sublayer_name in subset:
                print(f"Collecting sensitivity of module {sublayer_name} of layer {i}")
                collected_sensitivity[i][sublayer_name] = spqr_handler[sublayer_name].add_err_collect_sense(
                    percdamp=args.percdamp,
                    bits=args.wbits,
                    groupsize=args.groupsize,
                    sym=args.sym,
                    perchannel=args.perchannel,
                    round_zero=args.round_zero,
                    permutation_order=args.permutation_order
                )

    return collected_sensitivity
            

def get_sensitivity(model, args, device):
    tick = time.time()

    print("Loading data ...")
    dataloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model_path=args.model_path,
        seqlen=model.seqlen,
    )

    results = collect_sensitivity(model, dataloader, args, device)

    print(f"sensitivity collection time: {time.time() - tick:.1f}")
    return results

def main():
    import argparse

    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "model_path",
        type=str,
        help="path to llama model to load, as in LlamaForCausalLM.from_pretrained()",
    )
    parser.add_argument(
        "dataset",
        type=str,
        default="none",
        help="Dataset name [c4, pajama, refinedweb, none, etc.] or path to data where to extract calibration data from.",
    )
    parser.add_argument(
        "--custom_data_path",
        type=str,
        default=None,
        help="Path to load if specified. Deprecated",
    )
    parser.add_argument("--load", type=str, default=None, help="Path to load quantized statistics.")
    parser.add_argument("--save", type=str, default=False, help="Path to save quantized statistics.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--nearest", action="store_true", help="Whether to run the RTN baseline.")
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=None,
        help="How many weight columns (input features) are quantized with the same statistics, default = all of them",
    )
    parser.add_argument(
        "--permutation_order",
        type=str,
        default="identity",
        help="Weights permutation order; options: identity(default), spearman, act_order",
    )
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--new_eval",
        action="store_true",
        help="if this is set, evaluate on new (and slightly more realistic!) val dataset versions",
    )
    parser.add_argument("--sym", action="store_true", help="Symmetric quantization")
    parser.add_argument(
        "--perchannel",
        action="store_true",
        help="fit a unique quantizer to each output dim",
    )
    parser.add_argument(
        "--qq_scale_bits",
        type=int,
        default=None,
        help="Quantize quantization scale with this many bits (default=do not quantize)",
    )
    parser.add_argument(
        "--round_zero",
        type=int,
        default=None,
        help='whether to allow non-integer "zero" when quantizing weights non-symmetrically',
    )
    parser.add_argument(
        "--qq_zero_bits",
        type=int,
        default=None,
        help='Quantize quantization "zero" with this many bits (default=do not quantize)',
    )
    parser.add_argument(
        "--qq_zero_sym",
        action="store_true",
        help="enable sym=True in meta-quantization for groupwise zero, specifically",
    )
    parser.add_argument(
        "--qq_groupsize",
        type=int,
        default=16,
        help="Quantize quantization scale in groups of this many scales",
    )
    parser.add_argument(
        "--outlier_threshold",
        type=float,
        default=float("inf"),
        help="relative threshold for     outliers; higher threshold = more outliers.",
    )
    parser.add_argument(
        "--simplified_outliers",
        action="store_true",
        help="do not perform leave-one-out evaluation when detecting outliers; works faster, but generally worse in perplexity",
    )
    parser.add_argument("--wandb", action="store_true", help="Whether to use wandb or store locally.")
    parser.add_argument(
        "--skip_out_loss",
        action="store_true",
        help="Whether to skip computation of out loss.",
    )
    parser.add_argument(
        "--offload_activations",
        action="store_true",
        help="Offload activations to RAM to save GPU memory.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32"],
        help="dtype to load the model.",
    )

    args = parser.parse_args()

    if args.dataset == "custom":
        print(
            "WARNING: `--custom_data_path` argument and `--dataset=custom` option are DEPRECATED. ",
            "Pass dataset path directly to `dataset` argument or use 'pajama', 'refinedweb'",
            "See README.md for examples.",
        )
        args.dataset = args.custom_data_path

    if args.wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        args.exp_name = (
            os.environ.get("WANDB_NAME", "SpQR_run")
            + f"_wbits_{args.wbits}"
            + f"_groupsize_{args.groupsize}"
            + f"_qq_scale_bits_{args.qq_scale_bits}"
            + f"_qq_zero_bits_{args.qq_zero_bits}"
            + f"_qq_groupsize_{args.qq_groupsize}"
            + f"_outl_{args.outlier_threshold}"
            + f"_permord_{args.permutation_order}"
            + f"{'_new_eval' if args.new_eval else ''}"
        )
        wandb.init(
            config={a: getattr(args, a) for a in dir(args) if not a.startswith("_")},
        )
        wandb.run.log_code(".")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(args.model_path, args.load, args.dtype).train(False)
    results = get_sensitivity(model, args, device)

    df = pd.DataFrame(results)

    torch.save(df.to_dict(), 'collected_sensitivity.pt')


if __name__ == '__main__':
    main()