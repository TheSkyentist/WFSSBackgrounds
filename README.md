# WFSSBackgrounds
Create WFSSBackgrounds for JWST NIRISS WFSS Observations from available data. 

This repository contains scripts to create NIRISS WFSS Backgrounds from all available public data. These can be used while the NIRISS team works on creating improved backgrounds files for CRDS. Presently, the existing backgrounds leave systematic structure after subtraction, particularly noticeable closer to the ecliptic plane where the absolute background level is higher. 

The backgrounds created using this code are all available on [Zenodo](). If you find these backgrounds helpful or have any improvements you'd like to add, please let me know via email or pull-request. *If you use these in published scientific work, please credit this repository and the Zenodo Dataset.* 

## Usage

To use these scripts to create your own WFSS backgrounds, first make sure you have all of the dependencies. I have created a conda or mamba environment YAML file that should allow one to create an environment with the requisite packages quickly/easily.

First run the ```queryData.py``` script in order to download all of the public WFSS NIRISS RATE files. This script will exclude all products which have a 9.5mag or brighter star nearby. This downloading and cross-matching can take a while. The product lists will be saved in the created directories as well, along with a time-stamp of when the files were created. 

Then run the ```makeBackgrounds.py``` script to make all of the backgrounds from the downloaded files. The backgrounds are created by doing the following:
0. Flat-fielding the input data. This is done as these backgrounds are created with the purpose of being used with [grizli](https://github.com/gbrammer/grizli/) (hence the file-naming convention). Grizli expects flat-fielded WFSS Backgrounds, so by flat-fielding the inputs, we produce a flat-fielded background. Pass the "--dontFlat" command-line option to skip this. 
1. Measure a rudimentary background with sep. 
2. Perform source detection on the rudimentary sep-background-subtracted image. 
3. Mask all sources. 
4. Find the median of the (non-background subtracted) image (using sigma-clipping)
5. Divide each image by their respective medians. 
6. In each pixel, find the median value of all normalized images taking into account the source masks. 
7. Use a tophat kernel + convolution to fill in all remaining NaNs in the combined background. (This doesn't work 100% for grism observations with medium filters; since there are so few, the kernel size would need to be increased.)
8. Save the output. 

## Potential (Short-term) Improvements
1. **Better removal of completely contaminated images:** Currently I am only removing images which have a magnitude 9.5 star or bright nearby. However, this does not really check if the image is saturated, which can depend on the brightness of stars and the integration parameters of the image. However, the limit I invoke seems to work moderately well. 
2. **More refined source masking:** Currently I only do one pass of source masking. This is probably okay, but perhaps there are improvements with an iterative procedure? 
2. **Image smoothing:** I do not smooth the input or output images at all (save for some convolving to fill in NaN gaps, but this doesn't change any data that isn't NaN). Image smoothing might tamper down some of the variance in the final backgrounds. However, there are compact features (such as the occulting spots and their dispersed variants) that we don't want to smooth over.
