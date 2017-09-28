# 20170706

## [Narendra Ahuja](http://vision.ai.illinois.edu/ahuja.html) - Bird Watching

- Bird watching is a Low SNR, Low Sized detection

### Bird watching:
    - Stereo camera rectification of two images
        - Rectify using star points from RANSAC
    - Foreground detection
        - Now that we have aligned the stars, leave the stars behind, and only see the birds, or planes, etc.
    - Geometry verification
        - Improve SNR by seeing what remains consistent, or which is not like cosmic noise
    - Trajectory estimation
        - Now that we have bird points, compute the trajectory

### Image Super Resolution