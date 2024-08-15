#! /usr/bin/env sh

echo "Running all scripts"

# First Run
./queryData.py # Download Data
./makeArrays.py # This deletes the old applied directories and creates new source Masks
./makeBackground.py # Creates median background using naive masks
./denoiseNsmooth.py # Smooths the background
./applyBackgrounds.py # Applies the backgrounds
./createAppliedMask.py # Creates the source masks for the neI  subtracted images

# Iterative Run

# Iterate
for i in {1..2}
do 
./makeBackground.py # Creates the background using second round masks
./denoiseNsmooth.py # Smooths the background
./applyBackgrounds.py # Applies the backgrounds
./createAppliedMask.py # Creates the source masks for the new subtracted images
done

# Make plots
./compareBack.py # Compares the new background to the old background
./compareSub.py # Compares average subtraced profiles
./zodi.py # Plot the average background as a function of ecliptic latitude
./compareMask.py # Compare the masks on one image
./example.py # Example of the background artefacts