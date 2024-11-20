export MODEL_PATH='/raid/LLM/opt-125m'
export DATASET='pajama'

python collect_quant_loss.py $MODEL_PATH $DATASET \
    --wbits 4 \
    --groupsize 128 \
    --perchannel \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128 &&

python collect_quant_loss.py $MODEL_PATH $DATASET \
    --wbits 6 \
    --groupsize 128 \
    --perchannel \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128 &&

python collect_quant_loss.py $MODEL_PATH $DATASET \
    --wbits 8 \
    --groupsize 128 \
    --perchannel \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128 &&

python collect_quant_loss.py $MODEL_PATH $DATASET \
    --wbits 4 \
    --groupsize 32 \
    --perchannel \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128 &&

python collect_quant_loss.py $MODEL_PATH $DATASET \
    --wbits 6 \
    --groupsize 32 \
    --perchannel \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128 &&

python collect_quant_loss.py $MODEL_PATH $DATASET \
    --wbits 8 \
    --groupsize 32 \
    --perchannel \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128 &&


python collect_quant_loss.py $MODEL_PATH $DATASET \
    --wbits 4 \
    --groupsize 16 \
    --perchannel \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128 &&

python collect_quant_loss.py $MODEL_PATH $DATASET \
    --wbits 6 \
    --groupsize 16 \
    --perchannel \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128 &&

python collect_quant_loss.py $MODEL_PATH $DATASET \
    --wbits 8 \
    --groupsize 16 \
    --perchannel \
    --permutation_order act_order \
    --percdamp 1e0 \
    --nsamples 128