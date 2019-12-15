---
title: "Trends in Machine Learning: NeurIPS 2019 In Review"
date: "2019-12-15T23:40:37.121"
layout: post
draft: false
path: "/posts/ml-trends-neurips-2019/"
tags:
  - "Machine Learning"
  - "A.I."
  - "Deep Learning"
description: "Following an eventful week at NeurIPS 2019, I summarize some key trends in machine learning today."
---

<figure>
    <img src="./neurips_entrance.JPG" alt="neurips neural information processing systems 2019 entrance ">
</figure>

I had the great fortune of attending [NeurIPS 2019](https://nips.cc/) in Vancouver. NeurIPS is the main machine learning conference, and at over 13,000 attendees 🤯 it is arguably the largest in the field of artificial intelligence today. This year's edition of the conference consisted of a week packed full of workshops, talks, and posters ranging on everything from convergence bounds in neural networks to new benchmarks for natural language understanding. While admittedly a bit overwhelming in its size, there really was something for everyone at the conference and it's clear the excitement around machine learning is greater than it's ever been.

In this post, I want to distill some of the key topics in machine learning seen at the conference and include pointers to relevant papers. Such a list will of course be incomplete, and I highly encourage you to look through the complete conference proceedings for other work of interest. 

## Understanding the Dynamics of Neural Networks

Neural networks have been of tremendous practical interest for the last decade or so because of their superior performance on various tasks. However, the details of their learning dynamics and optimization behaviors are still quite poorly understood. Many papers at NeurIPS tried to shed some light on these questions.

In their work [Li and Wei et. al.](https://arxiv.org/pdf/1907.04595.pdf) analyzed the learning behavior of a small network trained with either a large learning rate that is annealed or a small learning rate from the start, showing that the small learning rate model first learns low-noise hard-to-fit patterns but then generalizes worse on higher-noise easier-to-fit patterns. [Levy and Duchi](https://arxiv.org/pdf/1909.10455.pdf) provided analyses on the gradient geometries that make stochastic and adaptive optimization methods optimal (though their analyses were done specifically for convex optimization scenarios). 

In their theoretical result, [Yun et. al.](https://arxiv.org/pdf/1810.07770.pdf) proved tight bounds on the number of hidden units that are necessary and sufficient for ReLu networks to memorize $N$ datapoints. While there has been recent work positing that neural networks can be learned whenever learning can be done with random features, [Yehudai and Shamir](https://arxiv.org/pdf/1904.00687.pdf) showed instead that this does not even hold in the conservative case of a single ReLu neuron.

It was very encouraging to see such papers attempting to reveal insight on the black box of neural network learning.

## Combatting Adversarial Examples

As powerful as neural networks are, some of their stark blind spots in feature space known as *adversarial examples* have been of serious interest for many years. This edition of NeurIPS showcased several elegant lines of work addressing some neural network shortcomings or shedding light on the adversarial example phenomena. 

One of the featured works from [Ilyas et. al.](https://arxiv.org/pdf/1905.02175.pdf) presented a fascinating construction showing a  mismatch between features that humans deem important for data learning tasks and those which models deem important. In particular, they demonstrated that adversarial examples may be due to the presence of *non-robust* features in several datasets which are highly predictive, and yet incomprehensible to humans. 

The work of [Yu and Hu et. al.](https://arxiv.org/pdf/1910.07629.pdf) devised a new defense against adversarial images in the setting where the adversary knows the detection mechanism, achieving 49% detection rate compared to 0% for previous methods. [Shafahi et. al.](https://arxiv.org/pdf/1904.12843.pdf) developed new algorithms for efficient adversarial training, allowing for 7-30x faster training of neural networks that are resilient to adversarial attacks.

## Secure and Private Machine Learning

Nowadays data privacy has surfaced to the forefront of societal consciousness, as people have become more sensitive to how their private data is being extracted and used. As a consequence, it is becoming increasingly important to develop models that can be securely and effectively trained while respecting data privacy concerns. The importance of this topic was evident in the sheer number of papers at NeurIPS contributing to this growing literature.

Works from [Ananda Suresh](https://arxiv.org/pdf/1910.03553.pdf), [Jordon et. al.](https://papers.nips.cc/paper/8684-differentially-private-bagging-improved-utility-and-cheaper-privacy-than-subsample-and-aggregate.pdf), and [Bernstein and Sheldon](https://arxiv.org/pdf/1910.13153.pdf) developed differentially private variants of anonymized histograms, model bagging algorithms, and Bayesian linear regression respectively. In a very neat privacy-preserving application, [De Cock et. al.](https://arxiv.org/pdf/1906.02325.pdf) developed a method for private classification of text messages that ensures the application doesn't learn anything about the messages and the text author doesn't learn anything about the application.

Several papers also exposed some issues with current work on machine learning privacy. For example, [Zhu et. al.](https://arxiv.org/pdf/1906.08935.pdf) showed that leaked gradients (previously thought to be safe to share) could be used to reconstruct private training data. [Bagdasaryan and Shmatikov](https://arxiv.org/pdf/1905.12101.pdf) demonstrated that models trained using differentially private SGD actually lead to drops in certain underrepresented dataset classes more than others, i.e. that unfair models become more unfair when trained using differential privacy.

While there is still a lot of work to be done in these privacy-preserving machine learning, there is great progress being made and issues being raised.

## Biology Meets Machine Learning

Today there is a growing interest in bringing ideas from related biological disciplines to machine learning. In his keynote talk, Yoshua Bengio proposed a theory for advancing to a new generation of machine learning theories. He advocated moving from what he called system 1 to system 2 deep learning using theories of consciousness from cognitive neuroscience which he called the ["consciousness prior"](https://arxiv.org/pdf/1709.08568.pdf). 

A number of papers at the conference dealt with the merging of neuroscience and deep learning. [Raghavan et. al.](https://arxiv.org/pdf/1906.01039.pdf) proposed an algorithm that could "grow" a multi-layered CNN-like neural network via self-organization and learning mechanisms modelled after the visual systems of animals days before they open their eyes. 

[Akrout et. al.](https://arxiv.org/pdf/1904.05391.pdf) described a new mechanism for the learning of appropriate synaptic weights that is more biologically plausible than the traditional feedforward path of an artificial neural network. [Zhang et. al.](https://arxiv.org/pdf/1908.06378.pdf) developed a variant of backpropagation that is more effective for training recurrent spiking neural networks.

In addition, there were a number of workshops trying to bring together biology and machine learning from different angles including the workshop on [Biological and Artificial RL](https://sites.google.com/view/biologicalandartificialrl/), the workshop on [Learning Meaningful Representations of Life](https://lmrl-bio.github.io/), and [Real Neurons and Hidden Units](https://sites.google.com/mila.quebec/neuroaiworkshop/).

## Applications Galore for Machine Learning

The conference also showcased a thriving plethora of machine learning applications. 

[Cardona et. al.](https://arxiv.org/pdf/1905.13290.pdf) developed generalizable techniques for predicting wind speeds from videos of naturally-occurring wind, with error approaching variability due to atmospheric turbulence. [Schumann and Lang et. al.](https://arxiv.org/pdf/1906.09621.pdf) tackled the problem of optimizing candidate hiring resource allocation, proposing new algorithms that outperform the status quo on real world data from a CS graduate program application.

[Ingraham et. al.](https://papers.nips.cc/paper/9711-generative-models-for-graph-based-protein-design.pdf) developed deep generative models to address the inverse protein folding problem, whereby protein sequences are generated from graph specifications of a desired target structure. [Subramanya et. al.](http://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf) created highly efficient techniques for doing accurate approximate nearest neighbor searches on a dataset of a billion points with a single reasonably-sized node.


## Final Thoughts

A week at NeurIPS showed that the machine learning world is as active as it's ever been. From theory to applications to everything in between, there is a lot of great work being done and many unsolved problems still left to be tackled. It will be exciting to see what the next year brings for this discipline!