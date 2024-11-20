export MODEL_PATH='/raid/LLM/opt-125m'
export DATASET='pajama'

python quant_sensitivity.py $MODEL_PATH $DATASET \
    --wbits 4 \
    --perchannel \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128