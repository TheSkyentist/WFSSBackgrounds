#! /usr/bin/env sh

echo "Running all scripts"

# First Run
./queryData.py # Download Data
./makeArrays.py # This deletes the old applied directories and creates new source Masks
./makeBackground.py # Creates median background using naive masks
./denoiseNsmooth.py # Smooths the background
./applyBackgrounds.py # Applies the backgrounds
./createAppliedMask.py # Creates the source masks for the new subtracted images

# Iterative Run
./makeBackground.py # Creates the background using second round masks
./denoiseNsmooth.py # Smooths the background
./applyBackgrounds.py # Applies the backgrounds
./createAppliedMask.py # Creates the source masks for the new subtracted images
./compareBack.py # Compares the new background to the old background
./compareSub.py # Compares average subtraced profiles

# Otherplots