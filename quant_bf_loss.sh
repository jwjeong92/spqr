export MODEL_PATH='/raid/LLM/opt-125m'
export DATASET='pajama'

python quant_bf.py $MODEL_PATH $DATASET \
    --wbits 6 \
    --perchannel \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128 \
    --ber 1e-3 \
    --seed 43 \
    --percentile 100 \
    --target_layer self_attn.v_proj fc1