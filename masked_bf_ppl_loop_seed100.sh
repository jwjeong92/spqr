export MODEL_PATH='/raid/LLM/opt-125m'
export DATASET='pajama'

seed=52
while [ $seed -le 142 ]
do
    i=1
    while [ $i -le 1 ]
    do
        BER=$(awk "BEGIN {print $i * 1e-3}")
        echo "Running with BER=$BER and seed=$seed"
        
        python masked_bf_ppl.py $MODEL_PATH $DATASET \
            --wbits 6 \
            --perchannel \
            --permutation_order act_order \
            --percdamp 1e0 \
            --nsamples 128 \
            --ber $BER \
            --seed $seed \
            --percentile 100
        i=$((i + 1))
    done
    seed=$((seed + 1))
done