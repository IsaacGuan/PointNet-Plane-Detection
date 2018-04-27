# PointNet-Plane-Detection

In this experiment, based on the implementation of [PointNet](https://github.com/charlesq34/pointnet), I tried to explore the potential of neural network classifiers for doing some more specific tasks on point clouds, e.g., 3D plane detection.

## Experimental Data

I chose 64 tables from the [ShapeNetPart](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html) dataset for training and 8 for testing, each of which has a significant planar surface. The picture below shows a part of the training set. For the training data, I marked out the points on the plane manually.

![training_data](training_data.jpg)

## Testing Result

I trained the network for 100 epochs and got a result of an accuracy around 85%. The plane detection result on the testing dataset is as below.

![testing_result](testing_result.jpg)

The result shows a few interesting patterns in it. The classifier seems to favor a table with a more normal shape, i.e., a table with a square tabletop and four straight legs. For tables without a regular shape, the classification accuracy is relatively lower, and the classifier tends to misclassify the points in the middle of tabletop.

For the very specific plane detection problem, such misclassification issue does not matter much, as we can apply a [3D Hough transformation](https://pdfs.semanticscholar.org/c52b/a582769375469d688e2a3b2d4d7e3f088472.pdf) afterwards on the detected planar part, which is robust towards missing and contaminated data, to generate the plane information.
