#!/bin/bash

start=$(date +%s)
parallel --jobs ${2} --colsep ',' --ungroup ./run_main.sh {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} :::: ${1}
end=$(date +%s)

