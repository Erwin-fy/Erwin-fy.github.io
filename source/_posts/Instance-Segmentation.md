---
title: Instance Segmentation
date: 2017-07-17 10:44:45
tags:
	- Deep Learning
	- Computer Vision
categories: 
	- cate
mathjax: true
---

# Instance-aware Semantic Segmentation via Multi-task Network Cascades
## 1. Overall Architeture(3 stages)
![MNC](MNC.png)

## 2.1	Stage1: Regressing Box-level Instances(class-agnostic)
<p>	__RPN network__(the same as the faster rcnn's) predicts bounding box locations(__4*9__) and objectness scores(__2*9__). 
    For the bbox $i$, let $B$<sub>i</sub> = {$x$<sub>i</sub>, $y$<sub>i</sub>, $w$<sub>i</sub>, $h$<sub>$i$</sub>, $p$<sub>i</sub>}, 
	__$L$<sub>1 i</sub> = SmoothL1Loss($x$<sub>i</sub>, $y$<sub>i</sub>, $w$<sub>i</sub>, $h$<sub>i</sub>) + SoftmaxLoss($p$<sub>i</sub>)__ </p>

## 2.2 Stage2: Regressing Mask-level Instances(class-agnostic)
<p> Generate 300 proposals from  Stage1's bboxes via __NMS__. Given the bboexs and feature map(Conv5\_3), RoI warping layer interpolates(bilinear) the features inside the bbox and outputs the features(28\*28), (We can get the weights relative to the pixel of Conv5\_3 at (u,v), then use bilinear interpolates). 
    A max pooling layer is then applied to produce a lower-resolution output(14\*14). 
    Following 2 fc layers generate m<sup>2</sup> outputs, each performing binary logistic regression to the ground truth mask(m=28, but it is 21 in caffe code??).
	__$L$<sub>2</sub> = SigmoidEntropyLoss(mask\_pred, gt)__ </p>


## 2.3 Stage3: Categorizing Instances
<p> Mask-base: Generate mask\_proposal by combining mask\_pred and proposal and resize it to RoI pooled resolution(14\*14). The masked feature is given by element-wise product: 
        $feature\_roi\_mask = mask\_proposal \* RoI Pooled$, followed by the 2 fc layers.
	Box-base: Append 2 fc layers to the RoI pooled feature.
	Concat 2 pathways, and outputs cls\_scores(21), seg\_cls\_scores(21) and bbox\_pred(84).
    __$L$<sub>3</sub> = SoftmaxLoss(cls\_scores) + SoftmaxLoss(seg\_cls\_scores) + SmoothL1Loss(bbox\_pred)__ </p>



## 3. Experiments 
<p> mAP<sup>r</sup>@0.5:63.5%
    mAP<sup>r</sup>@0.7:41.5% 
    time/img: 0.36s 
</p>

<p> According to the speed, it is slow. I think the main reason is the faster rcnn model is not fast enough, and there are some fc layers. 
    According to the mAP, due to cascades, if Stage 1 and 2 are not effective, the result will be not ideal.
</p>

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Mask R-CNN

## 1. Core Architecture
Faster R-CNN + RoIAlign + Mask-Branch

## 2. RoIPool RoIWarp RoIAlign
<p>	Faster R-CNN produces a conv feature map with several convolutional (conv) and max pooling layers(e.g., VGG16). After RPN network, original RoI regions are maped to this 	  conv layer, called proposals(r,c,w,h).

	RoIPool takes proposals as inputs and divides the h × w proposals(RoI) into an H × W grid of sub-windows of approximate size h/H × w/W. And then max-pooling or 		avg-pooling is applied to each sub-window. So the outputs are the feature maps with fixed size (H × W).

	Similar to RoIPool, RoIWarp has been discussed in MNC, adopting bilinear interpolation.

	Taking alignment into consideration, RoIAlign avoid any quantization. For example, if we want to tansform a 7\*7 RoI into 3\*3 feature map. How to process by using RoIPool? Let the size of sub-window is ***cell(7/3)*** and the sliding stride is ***floor(7/3)***, so these quantization cause misalignments.
	But with RoIAlign, we use ***(7/3)*** and compute via bilinear interpolation. </p>

## 3. Loss(Multi-Task)
<p>	__*L* = *L<sub>cls</sub>* + *L<sub>box</sub>* + *L<sub>mask</sub>*__
	
	__*L<sub>cls</sub>*__ and __*L<sub>box</sub>*__ are the same as Faster R-CNN's.
	The mask branch has a __K\*m\*m__ dimensional output and __*L<sub>mask</sub>*__ is only defined on the k-th mask(SigmoidEntropyLoss). </p>


----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Semantic Instance Segmentation via Deep Metric Learning
<p>This approach is different from the above two papers which are based on the __RCNN__, and it combines deep fully conv network and metric learning</p>

## 1. Deep Metric Learning
### 1.1 Metric Learning
<p>	Construct a distance function and compute similarity.
	[Refer to this blog](http://blog.csdn.net/nehemiah_li/article/details/44230053) </p>

### 1.2 Deep Metric Learning
<p>	In this paper, based on the deep fully conv embedding model, compute how likely two pixels are to belong to the same object and group similar pixels.  </p>

## 2. Overall Architecture
![DMetricL](DMetricL.png)

## 3. Model
### 3.1 Embedding vectors
<p>	Take as input a feature map and output a $[h,w,d]$ tensor. Thus each pixel $p$ in image is represented by $d$-dimensional embedding vector $e$<sub>p</sub>.
	Define the similarity between pixels $p$ and $q$ as Equation(1) and then define the loss function __*L<sub>e</sub>*__(similar to logistic regression)  </p>

### 3.2 Creating masks
<p>	Each pixel $p$ will generate a mask if picked as a seed, by finding all the other pixels $q$ that have a similarity with $p$ greater than a threshold $\tau$:
	__$mask(p, \tau)$ = \{ $q$: $\sigma$ $(p, q)$ $\geq$ $\tau$ \}__  
</p>


### 3.3 Classification
<p>	 The model also takes as input a feature map and outputs a $[h,w,C+1]$ tensor, predicting the class of each mask generated by each pixel. We can get __*L<sub>cls</sub>*__ 	using softmax cross-entropy loss. And the ground truth label is assigned to the pixel $p$ via IoU threshold between the proposed mask and ground truth masks. </p>

### 3.4 Seediness
<p>	It is obvious that the seediness tensor is computed from the classification tensor.
	Define the “seediness” of pixel p to be
	$S$<sub>p</sub> = $max$ $max$ $C$<sub>pc</sub><sup>$\tau$</sup>
	($C$<sub>pc</sub><sup>$\tau$</sup> represent the probability that pixel $p$ is a good seed for an instance class $c$ when using similarity threshold $\tau$.) </p>

### 3.5 Choose the seeds
<p>	According to 'Seediness' heatmap $S$<sub>p</sub> and Section 3.3 in the paper, we can choose good seeds. Then we will attain the mask and confidence score around the seeds with section 3.2, 3.3 and 3.4 </p>