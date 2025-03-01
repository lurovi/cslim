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
                --torus_dim "${11}" \
                --pop_shape "${12}" \
                --radius "${13}" \
                --cmp_rate "${14}" \
                --run_id "${15}"


end=$(date +%s)
