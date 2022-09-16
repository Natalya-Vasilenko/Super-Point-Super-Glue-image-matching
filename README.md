#  Super Point + Super Glue image matching for Lanit-Tercom dataset
## Tasks
To build a reconstruction of the car track from sequential photos of the road. The [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) model was chosen to work on extracting the key points of the images and matching them. The output data must be filtered by mask and converted to a format that supports [COLMAP](https://github.com/colmap/colmap).
## Contents
`main.py` : processes npz-files obtained with the [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) model for further reconstruction with COLMAP.
## Work
The npz-files for each image pair in the `\SuperGluePretrainedNetwork_match_pairs` were obtained using the `match_pairs.py` script from the [SuperGluePretrainedNetwork](https://github.com/magicleap/SuperGluePretrainedNetwork) repository.



As a result obtained in the `\keypoints` folder:
1. Files **{img_name}.jpg.txt**. Each of them starts with 2 integers giving the total number of keypoints and the length of the descriptor vector for each keypoint (128). Then the location of each keypoint in the image is specified by 4 floating point numbers giving subpixel column and row location, scale, and orientation. Finally, the invariant descriptor vector for the keypoint is given as a list of 128 integers in range [0,255].
2. File **npz_pairs.txt** with the names of the matched images and the indexes of the keypoints in the corresponding images.

