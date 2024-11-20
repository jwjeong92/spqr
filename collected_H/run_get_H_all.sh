export MODEL_PATH='/raid/LLM/opt-125m'
export DATASET='pajama'

python get_H.py $MODEL_PATH $DATASET \
    --permutation_order act_order \
    --percdamp 1e0 \
    --seed 0 \
    --nsamples 128 &&

python get_H.py $MODEL_PATH $DATASET \
    --permutation_order act_order \
    --percdamp 1e0 \
    --seed 42 \
    --nsamples 128 &&

python get_H.py $MODEL_PATH $DATASET \
    --permutation_order act_order \
    --percdamp 1e0 \
    --seed 43 \
    --nsamples 128 &&

python get_H.py $MODEL_PATH $DATASET \
    --permutation_order act_order \
    --percdamp 1e0 \
    --seed 44 \
    --nsamples 128 &&

python get_H.py $MODEL_PATH $DATASET \
    --permutation_order act_order \
    --percdamp 1e0 \
    --seed 45 \
    --nsamples 128 &&

python get_H.py $MODEL_PATH $DATASET \
    --permutation_order act_order \
    --percdamp 1e0 \
    --seed 46 \
    --nsamples 128 &&

python get_H.py $MODEL_PATH $DATASET \
    --permutation_order act_order \
    --percdamp 1e0 \
    --seed 47 \
    --nsamples 128 &&

python get_H.py $MODEL_PATH $DATASET \
    --permutation_order act_order \
    --percdamp 1e0 \
    --seed 48 \
    --nsamples 128 &&

python get_H.py $MODEL_PATH $DATASET \
    --permutation_order act_order \
    --percdamp 1e0 \
    --seed 49 \
    --nsamples 128 &&

python get_H.py $MODEL_PATH $DATASET \
    --permutation_order act_order \
    --percdamp 1e0 \
    --seed 50 \
    --nsamples 128 &&

python get_H.py $MODEL_PATH $DATASET \
    --permutation_order act_order \
    --percdamp 1e0 \
    --seed 51 \
    --nsamples 128