export MODEL_PATH='/raid/LLM/opt-6.7b'
export DATASET='pajama'

python masked_bf_ppl.py $MODEL_PATH $DATASET \
    --wbits 6 \
    --perchannel \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128 \
    --ber 0 \
    --seed 58 \
    --percentile 1