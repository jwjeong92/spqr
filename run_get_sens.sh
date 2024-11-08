export MODEL_PATH='/raid/LLM/opt-125m'
export DATASET='pajama'

python get_sensitivity.py $MODEL_PATH $DATASET \
    --wbits 3 \
    --perchannel \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128 