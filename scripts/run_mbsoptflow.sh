#!/bin/sh
DVCPROGRAM="~/mbsoptflow/mbsoptflow"
ANALYSISPROGRAM="~/mbsoptflow/voxel2mesh"
I0PATH="./volume0/slices_xyz/"
I1PATH="./volume1/slices_xyz/"
MASKPATH="./mask/slices_xyz/"
OUTPATH="./mbsoptflow/"
ARGS="-alpha 0.05 -level 12 -scale 0.8 -norm histogram_linear_mask -prefilter gaussian 1 --median -gpu0 0 --export_warp --conhull_maxz -localglobal 3 -transform1 sqrt --extrapolate --mosaic_approximation -overlap 100 --export_error --bone_quality"
$DVCPROGRAM -i0 $I0PATH -m $MASKPATH -i1 $I1PATH -o $OUTPATH $ARGS
