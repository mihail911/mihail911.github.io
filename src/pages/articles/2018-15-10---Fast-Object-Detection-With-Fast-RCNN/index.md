---
title: "Fast Object Detection with Fast R-CNN" 
date: "2018-10-01T23:46:37.121"
layout: post
draft: false
path: "/posts/better-object-detection-with-fast-rcnn/"
tags:
  - "Machine Learning"
  - "Computer Vision"
  - "Deep Learning"
description: "I dive into the details of Fast R-CNN, an extension to the original R-CNN model that boasted 9x speedup over its predecessor as well as state-of-the-art object detection results."
---

![Fast region proposal with convolutional neural networks cute dog cover](./dog_cover_photo.png)

In this article, we will continue in the vein of classic object detection papers to discuss **Fast R-CNN**. Continuing to 
study this line of region proposal with convolutional network work is rewarding because it allows us to see an iterative
refinement on a collection of models, each seeking to address shortcomings in its predecessor. This, after all, is how 
science should work and so the R-CNN saga is an elegant one to thoroughly understand.

Since this work is a follow-up to the original R-CNN paper, I will assume familiarity with the details of that model.
If you want a refresher on how the ancestor R-CNN worked, I highly recommend checking out [this post](/posts/object-detection-with-rcnn).

The Fast R-CNN work set out to address a number of problems in the original R-CNN including: 
1) Training R-CNN was a multi-state pipeline which made it a more unwieldy model to build and meant sufficient training data was needed at several steps 
2) Training was slow because certain steps involved writing to disk 
3) Object detection inference was quite slow (~47 seconds/image for certain models *even with a GPU*)

TLDR: 
1) Fast R-CNN created a single stage training pipeline making clever use of a multi-task loss function
2) No disk storage was needed for intermediate features because of the single-stage pipeline

## How Fast R-CNN Works

Let's start with our obligatory cute cat and dog photo: 

![Cute cat and dog best friends](./cat_and_dog.jpg)

Pause. That's really quite adorable. Ok, let's get back to what we're here to do.

Like the original R-CNN, the fast version also begins by extracting a set of around 2000 region proposals:

![Cat and dog region proposals extracted](./cat_and_dog_region_proposals.png)

Now for each region proposal, we run it through a  set of convolutional and max-pooling layers to extract a convolutional
feature map: 

![Region proposal through convolutional and max pooling layers to feature map](./region_proposal_through_conv_and_max_pool.png)

Now, given this feature map, we run a region-of-interest (RoI) through what is called an **RoI pooling layer**. This layer
takes an $h$ x $w$ RoI region and runs max-pooling across a grid of sub-regions within the RoI. The output is a fixed $H$ x $W$ feature map, 
where $H$ and $W$ are hyperparameters that are constant across all RoIs, regardless of dimension. 

After we have run our convolutional feature map through the RoI pooling layer, we are guaranteed a fixed-length output regardless
of region proposal size, so we can now execute a set of fully-connected layers to get an RoI feature vector. These
transforms look as follows:

![Convolutional feature map through RoI pooling layer and fully-connected layers](./conv_map_through_roi_pooling_and_fc.png)

Now we run that RoI feature vector through two sibling output layers: 1) a softmax classifier that outputs probabilities
for the $K$ object classes plus a background class and 2) a bounding box regressor that outputs refined bounding box positions for 
each of the $K$ object classes.

This looks as follows:

![Feature vector through softmax layer and bounding box regressor to get predictions](./softmax_and_bbox_regressor.png)

With these class probabilities and refined bounding box coordinates, we can output our final detection results for the original region proposal: 

![Final detection results for cat](./detected_cat.png)

Neat! And that is the essence of how the fast R-CNN model works. We'll dive into some details regarding how to build the model 
next.

## How Fast R-CNN is Built 

- describe multi-task loss function
- mention fine-tuning for detection
- mention truncated SVD for speedup?

## Experiments

- SOTA results
- fast training/testing times

- Ditch the SVMs (results slightly better without them)
- Easy on the proposal number

## Final Thoughts

