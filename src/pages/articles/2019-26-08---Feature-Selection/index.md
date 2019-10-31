---
title: "What Features Do You Want?"
date: "2019-08-25T23:46:37.121"
layout: post
draft: false
path: "/posts/feature-selection/"
tags:
  - "Machine Learning"
  - "A.I."
  - "Data Science"
description: "In which I discuss feature selection which is used in machine learning to help improve model generalization, reduce feature dimensionality, and do other useful things." 
---

<figure>
    <img src="./number-of-features-too-high.jpg" alt="feature selection meme">
</figure>

We are now going to start discussing another very important bit of practical machine learning
methodology: **feature selection**.

Feature selection is tightly related to model selection and
some might even argue that it is, in fact, a type of model selection. We won't get too hung up on the semantics,
and instead jump right into motivating why it's such an important technique.

## Motivations
        
Consider the following scenario: you are dealing with a supervised problem domain where you have come up with a
number of features that you deem could be important to building a powerful classifier.

To make the problem concrete,
let's say you are trying to build a spam classifier and you've come up with 10,000 features, where each feature
denotes whether a given email you are trying to classify contains (or doesn't) one of 10,000 possible words. So some of the
features could be **CONTAINS("the")**, **CONTAINS("cat")**, or **CONTAINS("viagra")**.

Let's say further that you have only
500 emails in your entire dataset. What's one **big** problem you may encounter?

Well, in this case, you have
far more features than you have data points. In such situations, your model will be overspecified. In other
words, you have a very **high chance of overfitting** on your dataset.

In fact, it's quite likely that most of the features
you've thought of won't actually help you build a more robust spam classifier. This is because you are
adding a feature for each of 10,000 words, and it seems unlikely that a feature like **CONTAINS("the")** will help you
determine a spam email.

Feature selection seeks to handle the problem of picking out some number of features that are the *most useful* for
actually building your classifier. These are the features that give you the most powerful signal for your problem.
So how do we do that?

## A First Pass at Selection

In turns out there are quite a few techniques commonly used for feature selection. We will describe a few of them.

The first selection method we will begin with is called **best-subset selection**. This selection method involves
trying out *every possible* subset of features from the entire collection.

Therefore if we had three possible features $(A, B, C)$, we
would consider the subsets: $(A)$, $(B)$, $(C)$, $(A, B)$, $(A, C)$, $(B, C)$, $(A, B, C)$. For each of these
feature subsets, we could run some sort of cross-validation technique to evaluate that feature set (for a review
of cross-validation check out the previous lesson).

The feature set which achieved the **lowest generalization error** would
be selected for our best-performing model. Here, lowest generalization error refers to the average error across the
folds of our cross-validation.

What's one issue that comes up with this technique? Well, notice how even
for a very small total feature set (3 in this case), we still had 7 feature subsets to evaluate.

In general, we can mathematically
demonstrate that for a feature set of size $N$, we would have $2^N - 1$ possible feature subsets to try. Techniqually
we have $2^N$ subsets, but we are disregarding the empty subset consisting of no features.

In other words, we have an **exponential number** of subsets to try! Not only is that a **HUGE** number of feature subsets even for a relatively small number, but on top of that, we have to run cross-validation on each feature subset.

This is prohibitively expensive from a computational standpoint. Imagine how intractable this technique would be with our 10,000 feature set for building
our spam classifier! So, what do we do?

## Improvements in Feature Selection

Notice that our best-subset feature selection technique guarantees finding the optimal feature set, since we are literally
trying every possible combination. This optimality comes at a price, namely waiting an impossibly long time for
finding the best subset on even small problems.

But what if we don't have to strictly achieve optimality? What if for our
purposes, all we need is *good enough*? **Forward-stepwise selection** helps us do just that: get good enough (Yay, C's get degrees).

How does it work? Forward step-wise selection basically functions as follows: Assume we have $k$ features in total to choose from. We
will order these features as $f_1$, $f_2$, ..., $f_k$, and let $S$ denote our best performing feature set so far. We execute the following procedure:

  1) Initially start with $S$ being the empty set (so no features).

  2) Repeat the following process: for each of the $k$ features we can choose from, let $S_k$ be the feature set with feature $f_k$ added to $S$. Run cross-validation using $S_k$. At the end assign $S$ to the best performing $S_k$ among all the $S_k$ created during the loop.

  3) Run the loop in 2) until our set <Bold txt="S"/> contains a number of features equal to some threshold we choose.

Hence, at the beginning $S$ is empty. Let's say we have three potential features $f_1$, $f_2$, and $f_3$. We
will run cross-validation three times, adding a single feature to $S$ each time. Hence we will have one run with the feature set $\{f_1\}$, one with $\{f_2\}$, and one with $\{f_3\}$.

The feature set with the best generalization error will be used for $S$. We then repeat this process, greedily adding a single
feature to our new $S$ each time, based on best generalization error.

And that's it! Notice that for this particular algorithm we can only run cross-validation
up to order $k^2$ times (can you see why?) This is clearly **MUCH** better than running cross-validation an exponential number
of times, which is what we get with best-subset selection.

There is another related feature selection technique called **backward-stepwise selection** where $S$ is initialized to
the set of all features, and then during each iteration of 2) above, we remove a feature instead of add a feature. The remainder
of the algorithm is the same as forward-stepwise selecion.

## Final Thoughts

Besides the aforementioned techniques, there are a number of more sophisticated feature selection
approaches. These include simmulated annealing and genetic algorithms, just to name a few. For now, we will leave discussion of those
techniques to another time! Regardless, feature selection is an important technique to add to the toolbox, as it enables us to be
more careful in how we build out our models.