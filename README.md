# Coupled-End-to-End-Transfer-Learning-With-Generalized-Fisher-Information

# Description

This is the implementation of coupled end-to-end transfer learning with generalized Fisher Information in domain adaptation from SVNH to MNIST.
In domain adaptation, we made teacher network and student network share the same encoder G. with different classifier C1, C2, and decoder D1, D2.
The teacher is trained to classify data in source domain and reconstruct target images. The student tried to classify source domain image and target images.
The student is also trained to replicate teacher's classification outputs as well as reconstruction results.




### Installation
- Install PyTorch (Works on Version 0.2.0_3) and dependencies from http://pytorch.org.
- Install Torch vision from the source.


### Train
For example, if you run an experiment on adaptation from svhn to mnist,
```
python main.py --source svhn --target mnist
``


# Citation

The related paper is published as below:

<pre><code>@InProceedings{Chen_2018_CVPR,
author = {Chen, Shixing and Zhang, Caojin and Dong, Ming},
title = {Coupled End-to-End Transfer Learning With Generalized Fisher Information},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}</code></pre>

[[PDF](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Coupled_End-to-End_Transfer_CVPR_2018_paper.pdf)] 
