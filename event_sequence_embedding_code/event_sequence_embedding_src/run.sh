#!/bin/bash

python -m src.train --file "C:\Users\tsun04\\event_sequence_embedding\\jrn_mobile_flatten.csv" \
                     --input_type "csv" \
                     --separator "," \
                     --folder "./output" \
                     --columns_to_select "Phrase" \
                     --size 50 \
                     --alpha 0.025 \
                     --window 5 \
                     --min_count 5 \
                     --max_vocab_size 100000 \
                     --sample 1e-3 \
                     --seed 1 \
                     --workers 4 \
                     --min_alpha 0.0001 \
                     --sg 0 \
                     --hs 0 \
                     --negative 10 \
                     --cbow_mean 1 \
                     --iter 5 \
                     --null_word 0