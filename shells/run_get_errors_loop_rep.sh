export MODEL_PATH='/raid/LLM/opt-125m'
export DATASET='pajama'

seed=42
while [ $seed -le 51 ]
do
    i=1
    while [ $i -le 10 ]
    do
        BER=$(awk "BEGIN {print $i * 1e-4}")
        echo "Running with BER=$BER and seed=$seed"
        
        python quant_bf.py $MODEL_PATH $DATASET \
            --wbits 6 \
            --perchannel \
            --permutation_order act_order \
            --percdamp 1e0 \
            --nsamples 128 \
            --ber $BER \
            --seed $seed \
            --percentile 100 \
            --bf_required
        i=$((i + 1))
    done
    seed=$((seed + 1))
done