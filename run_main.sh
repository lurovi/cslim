#!/bin/bash

start=$(date +%s)

python3 main.py --seed_index "${1}" \
                --algorithm "${2}" \
                --dataset "${3}" \
                --pop_size "${4}" \
                --n_iter "${5}" \
                --n_elites "${6}" \
                --pressure "${7}" \
                --slim_crossover "${8}" \
                --p_inflate "${9}" \
                --p_crossover "${10}" \
                --p_mutation "${11}" \
                --torus_dim "${12}" \
                --pop_shape "${13}" \
                --radius "${14}" \
                --cmp_rate "${15}" \
                --run_id "${16}"


end=$(date +%s)
