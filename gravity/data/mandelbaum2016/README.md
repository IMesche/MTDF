This directory contains data from [this paper from
2016](http://adsabs.harvard.edu/abs/2016MNRAS.457.3200M) about the galaxy halo mass vs. stellar mass
relation for locally brightest galaxies, split into red and blue subsamples.

The two subdirectories contain the lensing signals for LBGs and all Main sample galaxies,
respectively.  These data are shown in Figures 7, 8, and 9 of the paper.  In each subdirectory, the
file name indicates the color sample ("red" or "blue").  For LBGs, since the samples for blue
galaxies at high stellar mass were small, the data were shown rebinned for high stellar mass; the
file containing the rebinned data has "rebin" in the filename.

The data files contain a single column with the transverse separation in physical Mpc/h, followed by
pairs of columns that have DeltaSigma and its 1-sigma uncertainty (h Msun/(physical pc)^2).  The
stellar mass bins for those columns are given in the proper order in the header line.

Following agreement from collaborator Wenting Wang, who produced the LBG sample, the LBG sample used for this work may be downloaded from [here](https://cmu.box.com/s/d11e3tnaz0p0m83h64pserm9z9andiva).  When using this catalog, please reference both [Mandelbaum et al 2016](http://adsabs.harvard.edu/abs/2016MNRAS.457.3200M) and [Wang et al 2016](https://ui.adsabs.harvard.edu/abs/2016MNRAS.456.2301W/abstract).