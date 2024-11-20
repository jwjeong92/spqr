export MODEL_PATH='/raid/LLM/opt-125m'
export DATASET='pajama'

python get_H.py $MODEL_PATH $DATASET \
    --permutation_order act_order \
    --percdamp 1e0 \
    --seed 43 \
    --nsamples 128
