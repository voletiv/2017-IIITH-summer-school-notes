# 20170705

## [Gaurav Sharma](http://www.grvsharma.com/research.html) (IIT Kanpur) - Face and Action (14:30 - 15:30)

- (x_i - x_j)^T * L^T * L * (x_i - x_j) = ||(L*x_i - L*x_j)||^{2}_{2}

- We use L to project our input space into a space with better distance metrics for the semantics that matter, i.e. L*x_i

- Heterogeneous setting: some images have identity, some other images have tags, etc.

### PROPOSED METHOD

- Distance = Distance_common_across_tasks + Distance_specific_to_task

- During training, learn all tasks together -> update common projection for all tasks -> update projection for specific task

- Experimented with large datasets:
    - LFW
    - SECULAR - took images from Flickr, so hopefully no overlap with celebrity faces in LFW

- Comparable methods: WPCA, stML (single task), utML (union of tasks)

- Identity-based retrieval: (Main task, Auxiliary task)=(Identity, Age)

- Age-based retrieval: (Age, Identity)

- Also added expression information

- Adaptive LOMo

- Adaptive Scan (AdaScan) Pooling - CVPR 2017
