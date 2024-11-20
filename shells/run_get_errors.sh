export MODEL_PATH='/raid/LLM/opt-125m'
export DATASET='pajama'

python quant_bf.py $MODEL_PATH $DATASET \
    --wbits 8 \
    --perchannel \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128 \
    --ber 1e-4 \
    --seed 42 \
    --percentile 100 \
    --bf_required
