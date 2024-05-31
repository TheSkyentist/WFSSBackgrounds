# WFSSBackgrounds
Create WFSSBackgrounds for JWST NIRISS WFSS Observations from available data. 

This repository contains scripts to create NIRISS WFSS Backgrounds from all available public data. These can be used while the NIRISS team works on creating improved backgrounds files for CRDS. Presently, the existing backgrounds leave systematic structure after subtraction, particularly noticeable closer to the ecliptic plane where the absolute background level is higher. These backgrounds are also only for the full array, not subarray modes. 

The backgrounds created using this code are all available on [Zenodo](https://zenodo.org/records/10686452). If you find these backgrounds helpful or have any improvements you'd like to add, please let me know via email or pull-request. *If you use these in published scientific work, please credit this repository and the Zenodo Dataset.* 

**NOTE:** Backgrounds produced with this code should only be used for GR150C and GR150R observations with F115W, F150W, and F200W blocking filters. There are significant features in the other filters (especially the medium bands) and they have much less available data. 

## Usage

To use these scripts to create your own WFSS backgrounds, first make sure you have all of the dependencies. I have created a conda or mamba environment YAML file that should allow one to create an environment with the requisite packages quickly/easily.

First run the ```queryData.py``` script in order to download all of the public WFSS NIRISS RATE files. This script will exclude all products which have a 9.5mag or brighter star nearby. This downloading and cross-matching can take a while. The product lists will be saved in the created directories as well, along with a time-stamp of when the files were created. 

Then run the ```makeMask.py``` script in order to flat-field all of the downloaded data and mask the sources within. Even though flat-fielding is typically done after WFSSBackground subtraction in the JWST Spectroscopic Stage 2 Pipeline, it is necessary in this case to ensure compatibility with [grizli](https://github.com/gbrammer/grizli/). This also ensure that when we smooth the resultant median WFSSBackground, it does not inherit any structure from the flat-field. This script does the following:
1. Flat-fielding the input data with the JWST pipeline. 
2. Measure a rudimentary background with sep. 
3. Perform source detection on the sep-background-subtracted image. 
4. Mask all sources. 

Running the ```makeBackgrounds.py``` script will make backgrounds from the downloaded files. The backgrounds are created by doing the following: 
1. Find the median of the (source-masked) image (using sigma-clipping)
2. Divide each image by their respective medians. 
3. In each pixel, find the median value of all normalized images taking into account the source masks.

Finally, running 
1. Denoise the image using Non-Local Means, this smooths the image without losing the compact structures seen in the WFSSBackgrounds
2. Use [maskfill](https://github.com/dokkum/maskfill/tree/main) to fill in any remaining NaNs. 
3. Save the final backgrounds. In addition, one background is created with the flat-field reversed at the end, ensuring that these can be used with the JWST pipeline.

## Potential (Short-term) Improvements
1. **Better removal of completely contaminated images:** Currently I am only removing images which have a magnitude 9.5 star or bright nearby. However, this does not really check if the image is saturated, which can depend on the brightness of stars and the integration parameters of the image. However, the limit I invoke seems to work moderately well. 
2. **More refined source masking:** Currently I only do one pass of source masking. This is probably okay, but perhaps there are improvements with an iterative procedure? 
