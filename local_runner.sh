#!/bin/bash

# Arrays of options for each argument
ps=(32 64)
as=(8 16)
ls=(1e-3 1e-3)
tss=(251121222501 251121222502)

# Loop from 1 to 10, incrementing by 1
for (( i=1; i<=1; i++ )); do
    # Loop over each combination of arguments
    for ts in "${tss[@]}"; do
        for p in "${ps[@]}"; do
            for a in "${as[@]}"; do
                for l in "${ls[@]}"; do
                      # Randomly select a GPU between 0 and 8
                      gpu=$((RANDOM % 9))

                      # Run the Python script with the current combination of arguments in the background
                      python main.py -ts "$ts" -p "$p" -a "$a" -l "$l" -gpu "$gpu" &
                done
            done
        done
    done
done

wait