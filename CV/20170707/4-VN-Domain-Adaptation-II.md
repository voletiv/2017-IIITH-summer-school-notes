
## [Vinay Namboodiri](https://www.cse.iitk.ac.in/users/vinaypn/), IIT Kanpur - Domain Adaptation - II (15:00 to 16:30)

### Domain Adaptation for Detection

- Challenge is one does not know if object is present, and if it is present, where is it

- We want to align the subspaces of the bounding boxes

- Our approach:
    - Train R-CNN detector on source subspace from GT bounding box
    - Obtain bounding boxes on source by using detector (avoid non-maxima suppression)
        - This works better than the GT bounding boxes
    - Obtain predicted target bounding boxes using detector trained on source
    - Learn subspace alignment using the source and target bounding boxes
    - Project source samples onto target subspace using aligned subspace
    - Train the R-CNN detector again with the source samples projected onto the target subspace

### LSDA

### MMD regularization
