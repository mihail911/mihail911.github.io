---
title: "Fundamental Deep Learning Algorithms To Learn" 
date: "2018-08-14T23:46:37.121"
layout: post
draft: false
path: "/posts/fundamental-deep-learning-algorithms-to-learn/"
tags:
  - "A.I."
  - "Deep Learning"
  - "Machine Learning"
description: "A discussion of fundamental deep learning algorithms people new to the field should learn along with a recommended course of study."
---

When approaching deep learning for the first time, there is a huge difference between what I consider *foundational algorithms* (those that power just about every neural network model that has existed ever) and *architectures*.

I think this distinction is important because it will help you determine how best to *learn both*. I would argue the foundational algorithms are **more important to start with**, and they are a prerequisite for the architecture types.

What do I mean when I’m referring to *foundational algorithms*? These include, but are not limited to, the following:

- [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation). This algorithm is literally the engine that powers everything that a neural network is. Today there is **no deep learning without backpropagation**. It’s the elegant algorithm developed by Rumelhart, Hinton, and others [back in the 1980s](https://www.nature.com/articles/323533a0) that determines how we train models. For one of the most intuitive explanations of backprop I’ve encountered, [check out](http://cs231n.github.io/optimization-2/).
- [Gradient descent](https://en.wikipedia.org/wiki/Gradient_descent). This a **super important** algorithm for determining how we update weights of a neural network. Vanilla gradient descent forms the core of all the fancy other stuff you see in papers including [AdaGrad, Rmsprop, Adam, etc.](http://ruder.io/optimizing-gradient-descent/) so spend the time to learn it well. As a side note, though gradient descent is extensively used in deep learning, there’s nothing about the algorithm that restricts it to neural networks. In fact, it can be used for many different machine learning models including [linear regression](https://en.wikipedia.org/wiki/Linear_regression), [logistic regression](https://en.wikipedia.org/wiki/Logistic_regression), etc.

After you’ve got the foundational algorithms down, the model architectures refer to some of the model designs already mentioned including:

- [Feedforward networks](https://en.wikipedia.org/wiki/Feedforward_neural_network). The most vanilla flavor of neural network and important to understand.
- [Convolutional networks](https://en.wikipedia.org/wiki/Convolutional_neural_network). These networks form the core of all vision models.
- [Recurrent networks](https://en.wikipedia.org/wiki/Recurrent_neural_network). One of the go-to models for doing natural language tasks.

**Start with learning feedforward networks**, and then you can learn the other two architectures in whatever order makes most the sense for what you are working on.

Finally, a few other algorithms that are used extensively in neural networks which aren’t foundational, but are important to know for practical deep learning application. Learn these **after** the stuff above:

- [Dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks)). If you plan on using regularization for your neural network (and you inevitably will), this is the **most important regularization technique**. I have basically never built a model that didn’t use dropout.
- [Weight initialization schemes](http://cs231n.github.io/neural-networks-2/). It turns out when building neural networks, how you initialize your weights is **crucial** for determining whether or not the model trains successfully. Therefore a number of different heuristics have been developed for initialization that you should *learn eventually*.
