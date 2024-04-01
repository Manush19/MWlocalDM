# MWlocalDM
Study of the local distribution of DM in MW using MW-like galaxy catalogue

# Contains

1. Accessory (currently only in local machine): This folder consists of the main input data, which are (i) RotCurvs folder that contain galname_rotmod.dat files which stores the RC data from SPARC, (ii) sparc_dict.json containing the python dict of results from reliability study (the fit results of SPARC galaxies), (iii) mw_1.dat and mw_2.dat, the RC data of MW from PB14 (mw_1) and E18 (mw_2) shared by Aakash Pandey.

2. Notebooks/Part-1: The following .ipynb notebooks are there:
    (i) MWfid_analysis
    (ii) MWlike_selection
    (iii) MWRC_analysis
    
3. Notebooks/Part-2: This folder contains the .ipynb notebooks for analysis including a spherical bulge rather than exp bulge assumed before.

4. Output/figures: for storing figures

5. Output/Ultra: ultraroot for ultranest fits (general)

6. MW_dict.pkl: python dict containing the fiducial ranges and other details of MW 

7. MWlike_dict.pkl: python dict for storing details of MW-like galaxy catalogue
