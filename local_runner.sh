#!/bin/bash

# Loop from 1 to 4, incrementing by 1
for (( i=1; i<=4; i++ )); do
    # Randomly select a GPU between 0 and 8
    gpu=$((RANDOM % 9))
    # Run the Python script with the current combination of arguments in the background
    python main.py -r "$i" -gpu "$gpu" &
done

wait