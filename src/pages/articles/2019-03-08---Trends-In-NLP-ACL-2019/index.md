---
title: "Trends in Natural Language Processing: ACL 2019 In Review"
date: "2019-08-03T23:40:37.121"
layout: post
draft: false
path: "/posts/nlp-trends-acl-2019/"
tags:
  - "Natural Language Processing"
  - "A.I."
  - "Deep Learning"
description: "Following an eventful week at ACL 2019, I summarize key trends and takeaways to look for in NLP."
---

<figure>
    <img src="./acl_2019_entrance.JPG" alt="entrance to ACL 2019">
</figure>

This week I had the great fortune to attend the [Annual Meeting of the Association for Computational Linguistics (ACL) 2019](http://www.acl2019.org/EN/index.xhtml) held in wonderful Florence in an old Medici family fortress.

Conferences are some of my favorite events to attend because in a very short amount of time you are able to tap into the stream-of-consciousness of a community, to learn what people are thinking and where the field is going. Given that ACL is arguably the largest gathering of NLP researchers from around the world, it offers a representative sampling of the community's headspace. 

As with other conferences, at times attending ACL was like drinking water from a fire hydrant, where you are getting overwhelmed with papers, presentations, and ideas. In this post, I want to distill some of the key learnings and trends I have gathered from a week spent with the NLP community, what the state of the field is in 2019 and where it is heading. When appropriate, I will reference papers highlighting some of these trends. Such a reference list will necessarily be incomplete, so I encourage you to look at the [full conference proceedings](http://www.acl2019.org/EN/program.xhtml) to see the other great work presented. 


## Reducing Bias in NLP

In his opening address, ACL president [Ming Zhou](https://www.microsoft.com/en-us/research/people/mingzhou/) noted that this year's ACL was the largest conference in the event's history with more than 2900 submissions, an increase of 75% over 2018! The field of natural language processing is on fire🔥 with academic and industry enthusiasm at record highs. 

However, these statistics on popularity give a fairly inadequate representation of global NLP interest. The bulk of this growth is happening in Northern America (led by the U.S.) and Asia (led by China), leaving the bulk of South America and Africa as well as many countries in Europe behind. As a consequence, the current state of NLP runs the risk of suffering from geographic bias where we don't get a diverse set of opinions and perspectives to sculpt the future of the field. 

As Ming Zhou noted (drawing on his experience growing the NLP community presence in the Asia Pacific region), one solution may be to hold more conferences and events in the underrepresented regions, as doing so in the past has historically led to more memberships in the host areas. There are already a number of great ongoing efforts to tackle the problem in this fashion such as the [Deep Learning Indaba](http://www.deeplearningindaba.com/).

Outside of geographic bias, there is also an increasing awareness of other unfortunate artifacts in current natural language processing development such as gender bias. Several papers underscored these facts through empirical studies. For example, [Stanovsky et. al.](https://arxiv.org/abs/1906.00591) demonstrated that four industrial machine translation systems as well as two current academic state-of-the-art (SOTA) models are significantly prone to gender-based translation errors. 

The community is also well-aware of this problem, and therefore a number of interesting works have been proposed such as that of [Kaneko et. al.](https://arxiv.org/pdf/1906.00742.pdf) which developed a debiasing method for word embeddings that preserves non-discriminative gender-related information while removing stereotypical gender biases. At a more high-level, this year's ACL also hosted the first [Gender Bias in NLP Workshop](https://genderbiasnlp.talp.cat/) and [Widening NLP Workshop](http://www.winlp.org/winlp-2019-workshop/) which are great efforts to bring together researchers working on these problems, to raise further awareness, and to stimulate fruitful discussion. 

There is still a lot of work to do certainly, but it is encouraging to see the community taking active steps to alleviate the bias problem.

## NLP Applications Galore

The current state of natural language processing is exciting among other reasons because our field is now in a position where the models and tools we develop have the potential to address many practical problems. This became clear when looking at the diverse collection of NLP applications showcased at the conference. 

In an era of fake news and fake neural news, verifying the truthfulness of statements is becoming increasingly different. The work of [Shengli Hu](https://www.aclweb.org/anthology/P19-1039) built a system using acoustic and linguistic features to identify concealed information from text and speech, outperforming humans by over 15%! 

In the health domain, the work of [Shardlow et. al.](https://www.aclweb.org/anthology/P19-1037) developed a neural model for making clinical letters written by doctors more readable for patients via domain-specific phrase tables. On a related line, [Du et. al](https://arxiv.org/pdf/1906.02239.pdf) proposed the task of symptom extraction from clinical conversations along with baseline models, as a means of reducing the time primary care physicians must spend on interacting with clinical documentation systems.

This year's ACL also featured an [entire workshop](https://aclweb.org/aclwiki/BioNLP_Workshop) dedicated to NLP applied to problems in biology. For example, [Fauqueur et. al.](https://arxiv.org/pdf/1907.01417.pdf) presented techniques for extracting new facts from biomedical literature without requiring training data or hand-crafted rules. Another neat paper was that of [Rajagopal and Vyas et. al.](https://www.aclweb.org/anthology/W19-5009) which adapted semantic role labelling systems to biological processes by pretraining an LSTM-CRF model on a large dataset and then finetuning on a low-resource corpus, achieving an improvement of 21 F1 points on a standard dataset!

Other cool applications of NLP include the work of [Zhang et. al.](https://arxiv.org/abs/1906.03497) which introduced the problem of email subject line generation (think Gmail Smart Reply but for email title generation), demonstrating a promising first model for the problem according to automatic and human evaluations. 


## Pretrain then Finetune: A New Paradigm for NLP 

Just as neural networks suddenly revolutionized the field of computer vision in 2011 with the seminal work of [Krizhevsky et. al.](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf), the story of deep learning for natural language processing has similarly been one of explosive and rapid growth. 

From 2015 - 2017, most tasks in NLP could be tackled via a relatively straightforward formula: embed textual input via some sort of continuous vector representations, encode these representations, attend to your encoded representations, and then predict for your task. This formalism is described nicely in a [post by Matthew Honnibal](https://explosion.ai/blog/deep-learning-formula-nlp). 

While conceptually simple, the *embed, encode, attend, predict* formula seemed virtually unstoppable, achieving SOTA results on all types of tasks such as machine translation, question answering, and natural language inference just to name a few. For a while, it seemed no problem was immune. 

Nowadays, it's fair to say there is a new sheriff in town. With the advent of powerful pretrained representations, trained using some flavor of a language modelling objective such as [ELMO](https://arxiv.org/abs/1802.05365), [OpenAI GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf), and [BERT](https://arxiv.org/pdf/1810.04805.pdf), the de facto technique for NLP has become to take some sort of off-the-shelf model pretrained on **gargantuan** amounts of data and fine-tune to your task with some smaller in-domain corpus. Indeed, this strategy has successfully achieved tremendous SOTA results on existing NLP benchmarks. 

At ACL, the dominance of this strategy was reinforced by some of the published work as well as people's general attitude toward the state of NLP. One line of work by [Dai and Yang et. al.](https://arxiv.org/pdf/1901.02860.pdf) sought to push transformer-based language super-models even further, improving their speed tremendously and achieving SOTA perplexity numbers. Another very representative work of this new paradigm was that of [Liu and He et. al.](https://arxiv.org/pdf/1901.11504.pdf) which leveraged a BERT-based architecture to top the GLUE benchmark leaderboard (at the time of submission).

Besides these works themselves, the general talk around the conference was that many architectures presented could achieve a few percentage points of improvement if they used something like BERT. The question then becomes: has this new paradigm trivialized many of the modelling innovations in NLP? 

My personal belief is **no**. In general, there are still *many* lines of work that are still very underexplored and will be crucial for pushing forward the next iteration of NLP progress. I outline a few below.


## Infusing Knowledge into NLP Architectures

While the existing pretrained language super-model architectures are very powerful, the manner in which they are trained from raw textual corpora encourages a *you get what you get* flavor of learning. In other words, what they learn is fairly unconstrained and their superior performance could be a function of just seeing many instances of textual sequences in various contexts from the huge datasets on which they are trained. Can we move beyond this by infusing information from grounded knowledge sources?

There were a number of papers trying tackle this question at ACL. For example, [Zhang et. al.](https://arxiv.org/pdf/1905.07129.pdf) used typed entity embeddings and alignments to an underlying knowledge graph to enhance BERT representations, showing that their model can outperform BERT on entity typing and relation classification. [Yang et. al](https://www.aclweb.org/anthology/P19-1226) also tackled this problem by proposing KT-NET, which uses an attention mechanism to fuse selected information from knowledge bases such as WordNet and NELL, thereby achieving SOTA on Squad 1.1 (at the time of submission). Another nice paper was that of [Logan et. al.](https://arxiv.org/pdf/1906.07241.pdf) which proposed the knowledge graph language model, a generative architecture that selectively copies facts from a knowledge graph relevant to an underlying context, outperforming strong baseline language models. 

While infusing knowledge into neural models is certainly a difficult problem, the results seem very promising!

## Interpretability of Models

Neural networks are known to be black-box models, for which it is particularly difficult to truly understand decision functions learned. Putting aside considerations of whether complete interpretability of these models is strictly necessary, it is fair to say that some level of understanding of model internals could greatly inform future architecture designs. Several nice papers at ACL aimed to shine some light on existing models.

The work of [Serrano et. al.](https://arxiv.org/pdf/1906.03731.pdf) elegantly challenged the folk wisdom that attention mechanisms indicate model notions of importance, showing that while sometimes this is true, there are instances where alternative ranking metrics may be more effective in indicating model decision processes. 

In another line of inquiry, [Jawahar et. al.](https://hal.inria.fr/hal-02131630/document) probed into the language structure learned by BERT, demonstrating that BERT's layers learn rich linguistic information such as surface features at the bottom layers, syntactic features in the middle, and semantic features at the top. The authors further suggest that deeper layers are necessary to learn long-distance dependency information.

Other works also tackled model explainability. [Gehrmann et. al](https://arxiv.org/pdf/1906.04043.pdf) developed a tool for detecting neurally-generated fake text by visualizing model densities of predicted word tokens, allowing human users to improve their detection rate by **nearly 20%**. [Sydorova et. al.](https://arxiv.org/pdf/1906.10924.pdf) investigated a number of post-hoc explanation methods such as [LIME](https://github.com/marcotcr/lime) on question-answering systems, demonstrating that certain techniques can help humans identify the superior QA model out of several options.


## Rethinking Evaluation and Assumptions of Natural Language Generation

As someone that works on dialogue systems, I have a soft-spot for the complexities of the natural language generation task. In particular, the notion of evaluation is still a very contentious issue for the community, so it was encouraging to see active work on improving the status quo. 

For starters, [Maxime Peyrard](https://www.aclweb.org/anthology/P19-1502) demonstrated that certain automatic evaluation metrics for summarization are inconsistent when it comes to assessing performance in certain scoring ranges. [Clark et. al.](https://homes.cs.washington.edu/~nasmith/papers/clark+celikyilmaz+smith.acl19.pdf) then proposed a new evaluation metric for generated text based on sentence mover's similarity, which is shown to correlate better with human judgment than the standard [ROUGE](http://www.aclweb.org/anthology/W04-1013). 

Text generated by models tends to suffer from factual errors and spurious statements. As a consequence, [Falke et. al.](https://leoribeiro.github.io/papers/acl19-summary-correctness.pdf) investigated whether natural language inference systems could be used for reranking outputs as a means of dealing with this issue. They found that out-of-the-box NLI systems aren't yet good enough to adapt to downstream tasks and offered tools for getting these systems to their necessary performance.

The more fundamental work of [Maxime Peyrard](https://www.aclweb.org/anthology/P19-1101) developed theoretically rigorous definitions of certain concepts in summarization such as redundancy, relevance, and informativeness. 

Outside of evaluation, the wonderful work of [Sankar et. al.](https://arxiv.org/abs/1906.01603) called into question certain assumptions about what traditional recurrent and transformer-based sequence-to-sequence dialogue models learn from a conversation history. In particular, they showed that these models are not very sensitive to certain perturbations applied to a context, challenging the efficacy of dialogue natural language generators. 

## Going Beyond the Pretrain-Finetune Paradigm

While the current iteration of NLP models has blown the SOTA out of the water, the prevailing sentiment in the NLP community is that there is still something left to be desired. We are in a bit of a tricky situation because we often use benchmarks to gauge progress on tasks, and many of these models are nearing or surpassing human performance on these existing NLP benchmarks. So where does that leave us?

This was the question posed by [Zellers et. al.](https://arxiv.org/pdf/1905.07830.pdf) which in earlier work had introduced a challenge dataset for the problem of commonsense natural language inference, only to find soon after the release that BERT had attained near human performance. To combat this, the authors proposed a follow-up dataset that was developed using a technique called adversarial filtering to select examples that BERT and other models would find difficult to answer. In the process, they drastically increased the complexity of the benchmark. 

BERT is certainly not perfect. A study by [Nangia et. al.](https://arxiv.org/pdf/1905.10425.pdf) showed that BERT-based models struggle with low-resource sentence-classification tasks and proposed a follow-up natural language understanding benchmark called [SuperGLUE](https://arxiv.org/pdf/1905.00537.pdf) that specifically focuses on evaluating such regimes.

Another line of work from [McCoy et. al.](https://arxiv.org/pdf/1902.01007.pdf) demonstrated that BERT models applied to natural language inference are learning very simple syntactic heuristics that do not generalize well to other entailment examples. They also release an evaluation set to determine whether models are adopting these heuristics but not solving the more general inference problem. 

A related paper from [Min and Wallace et. al.](https://arxiv.org/pdf/1906.02900.pdf) showed that many of the models proposed for the HotpotQA dataset, a multi-hop question-answering benchmark, do not actually need to perform multi-hop reasoning to get good performance. 

When all is said and done, in general my feeling is that the bulk of models today are still solving datasets rather than the tasks. We are building models that have become surprisingly effective at picking up and exploiting dataset-specific biases. In the process, our evaluation metrics paint fairly misleading pictures. This reminds me of Goodhart's law: *When a measure becomes a target, it ceases to be a good measure*. So how do we move forward?

Given that these evaluation benchmarks are proxies for natural language tasks and given the rapid pace of model development, it seems unreasonable to assume that benchmarks can remain static. Instead what I find to be especially promising is development of a suite of evolving benchmarks of increasing difficulty that each push the goalpost on natural language abilities further. Perhaps in the limit this is how we can achieve human-level NLP capabilities in machines.

## Final Thoughts

To wrap up, my week of ACL showed that the field of NLP is thriving! The community is experiencing very exciting times, and there are a lot of promising developments on the horizon. While the past year has seen substantial progress in the community, there are still *many* salient challenges and open problems to solve. 