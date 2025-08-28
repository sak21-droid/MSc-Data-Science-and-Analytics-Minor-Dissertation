# MSc-Data-Science-and-Analytics-Minor-Dissertation
The following repository contains code for all analyses of the three wave clips used for the project. Python was used for all visualizations and analysis of large-scale wave clips(Wave Clip 1 and Wave Clip 2), while R was used for full PCA and reconstruction of the small-scale wave dataset. The code uses the methodology outlined in the project and creates plots seen throughout the project.
All datasets and reconstructed results can be accessed by clicking the link below : 
https://uccireland-my.sharepoint.com/:f:/g/personal/f_osullivan_ucc_ie/EsJ56stvFKZAitrA6IC8hIQB2ZVhaCtfXcU8agcScrNaIg?e=jCro1a



Contents

Wave Clip 1 Analysis.py
Python script performing PCA analysis and reconstruction of Wave Clip 1. 

Wave Clip 2 Analysis.py
Python script performing PCA analysis and reconstruction of Wave Clip 2. Includes similar outputs as Wave Clip 1 analysis.

Small Scale Wave Clip Plots.py
Python script for generating visualizations (plots) for small-scale wave video datasets. Useful for inspecting temporal and spatial PCA modes on a smaller dataset.

Small Scale Clip Analysis.R
R script performing PCA-based analysis and reconstruction on small-scale wave video clips. Includes frame and video reconstruction 

UCC Masters Dissertation Saksham Kheter Pal.pdf
PDF version of the MSc thesis documenting the full methodology, theoretical background, experiments, results, and discussion.

Requirements
Python Scripts

Python 3.10+

Packages: numpy, opencv-python, matplotlib, scikit-learn, imageio, av

R Script

R 4.3+

Packages: av, magick, png, irlba, ggplot2

Instructions

Ensure all required packages are installed.

update the file paths in the code.


Notes

Scripts have been tested on medium-scale video datasets; large videos may require more memory and computation time.

All visualizations and reconstructed videos are reproducible from the provided scrip

