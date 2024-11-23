export MODEL_PATH='/raid/LLM/opt-6.7b'
export DATASET='pajama'

python get_H.py $MODEL_PATH $DATASET \
    --permutation_order act_order \
    --percdamp 1e0 \
    --seed 0 \
    --nsamples 128
