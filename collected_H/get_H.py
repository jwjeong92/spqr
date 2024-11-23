import os
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  # Set the GPU 2 to use

import torch
import torch.nn as nn
from tqdm import trange, tqdm

import pandas as pd

from spqr.modelutils import get_model
from datasets import load_dataset
from spqr.calibutils import get_hessian_matrix

try:
    import safetensors  # noqa: F401

    has_safetensors = True
except ModuleNotFoundError:
    has_safetensors = False

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
    parser.add_argument("--save", type=str, default=False, help="Path to save quantized statistics.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration data samples.")
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--permutation_order",
        type=str,
        default="identity",
        help="Weights permutation order; options: identity(default), spearman, act_order",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(args.model_path).train(False)
    
    # get_hessian_matrix는 perm, dead, ordered H_inv_diag를 반환함
    results = get_hessian_matrix(model, args, device)

    df = pd.DataFrame(results)

    model_name = args.model_path.split('/')[-1]

    results_name = f'H_{model_name}_{args.dataset}_seed{args.seed}.pt'
    folder_name = '/home/jwjeong/workspace/spqr_sens/collected_H'
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Directory Created: {folder_name}")
    else:
        print(f'Directory already exists: {folder_name}')
    
    torch.save(df.to_dict(), f'{folder_name}/{results_name}')
    
if __name__ == '__main__':
    main()