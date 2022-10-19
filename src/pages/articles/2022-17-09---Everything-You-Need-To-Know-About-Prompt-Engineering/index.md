---
title: "Everything You Need to Know About Prompt Engineering For Large Language Models"
date: "2022-09-16T23:40:37.121"
layout: post
draft: true
path: "/posts/a-complete-introduction-to-prompt-engineering/"
tags:
  - "Data Science"
  - "Machine Learning"
  - "NLP"
description: "I provide a complete introduction to prompt engineering for large language models."
---

<figure>
    <img src="./brown_puzzle.jpg" alt="half finished puzzle">
</figure>

In recent years, with the release of large language models (LLMs) pretrained on massive text corpora, a new paradigm for building natural language processing systems has emerged. 

Rather than the conventional methodology of building text applications that has been used for decades and relies on a carefully curated, labelled training set, LLMs have birthed a new technique called prompting. 

In the prompting paradigm, a pretrained LLM is provided a snippet of text as an input and is expected to provide a relevant completion of this input. These prefixes may describe a task being asked of the model such as:
```
Translate the following sentence from English to Spanish.

The cat jumped over the moon.
```

and the model is expected to return:
``` 
El gato saltó por encima de la luna.
```

The extraordinary thing about prompting is that if these inputs are appropriately crafted, a single LLM can be adapted to scores of diverse tasks such as summarization, question answering, SQL generation, and translation with a handful or zero training samples! 

Because the performance of these LLMs is so dependent on the inputs fed into them, researchers and industry practitioners have developed the discipline of **prompt engineering** which is intended to provided a set of principles and techniques for designing prompts to squeeze out the best performance from these machine learning behemoths. 

For this reason, prompt engineering is also sometimes called prompt programming or even [natural language programming](https://generative.ink/posts/methods-of-prompt-programming/).

In this post, I will provide a comprehensive review of the most interesting research, techniques, and use-cases in prompt engineering as applied to large language models. My goal is also to provide a set of actionable steps for being a more effective prompt engineer. 

Let's begin.

TODO: Provide a TLDR of practical prompt engineering tips/tricks (using Monica Lent-style template).


## Background on Prompt Engineering
  - Why do we care? Prompting demonstrated that this technique can be applied to LLMs but due to resource constraints of most labs, many efforts now devoted to exploring the same phenomena in smaller LLMs such as T5, BART, GPT2, etc. [Emergent abilities for LLMs](https://arxiv.org/pdf/2206.07682.pdf)

## Principles for Prompt Engineering

Prompting for large language models typically takes one of two forms: few-shot and zero-shot. In the few-shot setting, a translation prompt may be phrased as follows:
```
Translate from English to Spanish.
English: I like cats.
Spanish: Me gustan los gatos.

English: I went on a trip to the bahamas.
Spanish: Fui de viaje a las bahamas.

English: Tell me your biggest fear.
Spanish:
```
where the important thing to note is that the prompt includes a handful of examples showing how to perform the task of interest correctly (TODO: include reference here). Various research claims that by providing these demonstrations, the LLM is *learning* how to perform the task on the fly.

In the zero-shot setting, no examples are provided in the prompt as the translation task may be formulated as follows:

```
Translate the following sentence from English to Spanish.

The cat jumped over the moon.
```

Various research has shown intriguing prompt phenomena in LLMs. 

For example, [Lu et al.](https://arxiv.org/pdf/2104.08786.pdf) observed that in the few-shot setting, the order in which examples are provided in the prompt can make the difference between near state-of-the-art and random guess performance. This phenomenon is agnostic to the large language model size (i.e. larger models suffer from the same problem) and the subset of examples used for the demonstration (i.e. more examples in the prompt doesn't reduce variance). In addition, the performance of a given example ordering doesn't translate across model types. They then propose an entropy-based probing technique to generate the optimal prompt ordering without a development dataset. The approach is shown to robustly reduce variance for models even across diverse prompt templates. 

<figure>
	<img src="./order_sensitivity.png" alt="Order sensitivity for different training samples in large language models">
	<figcaption>Performance of different LLMs as we increase the number of training samples in the prompt, demonstrating high variance across model sizes.</figcaption>
</figure>

[Zhao and Wallace et al.](https://arxiv.org/pdf/2102.09690.pdf) also do an in-depth study of the instability of few-shot prompting. They show that when few-shot prompting, LLMs suffer from three types of biases: 

- Majority label bias - they tend to predict training samples that appear frequently.
- Recency bias - they tend to predict answers near the end of the prompt.
- Common token bias - they tend to predict answers that appear frequently in the training data.

They then describe a calibration technique designed to mitigate some of these biases, showing a reduction in variance and an up to 30% absolute accuracy bump.

<figure>
	<img src="./calibration_plot.png" alt="Calibration plot for different training samples in large language models">
	<figcaption>Performance of different-sized LLMs depending on whether the models are calibrated to address biases or not.</figcaption>
</figure>

  - reframing instructional prompts in GPTK's language
  - https://blog.andrewcantino.com/blog/2021/04/21/prompt-engineering-tips-and-tricks/
    - The LLM is completing a document, and documents rarely change writing style halfway through.
    - Consider dynamically selecting the most relevant few-shots
    - instead of generating N list items, generated 1 list item N times
    - generate many samples and rank them

Other research by [Reynolds and McDonell](https://arxiv.org/pdf/2102.07350.pdf) makes the claim that *few-shot learning* is actually a misnomer and, in fact, LLMs use few-shot examples to locate an appropriate task in an existing space of tasks learned from the training data. This further justifies the need for really carefully-designed prompt engineering. Thus, they propose a few principles that should be employed when prompting:
- Use declarative and direct signifiers for tasks such as *translate* or *rephrase this paragraph so that a 2nd grader can understand it*.
- Use few-shot demonstrations when the task requires a bespoke format, recognizing that few-shot examples may be interpreted as a whole by the model rather than independent samples.
- Specify tasks using characters or characteristic situations as a proxy for an intention such as asking Gandhi or Nietzsche to solve a task. Here you are tappin g into LLMs' sophisticated understanding of analogies
- Constrain the possible completion output using careful syntactic and lexical prompt formulations such as saying *Translate this French **sentence** to English*  or adding quotes around the French sentence.
- Encourage the model to break down problems into sub problems via step-by-step reasoning.

[Andrew Cantino](https://blog.andrewcantino.com/blog/2021/04/21/prompt-engineering-tips-and-tricks/) also provided a handful of practical tips and tricks for prompt engineering. These include:
- Make sure your inputs are grammatically correct and have good writing quality as LLMs tend to preserve stylistic consistency in their completions.
- Rather than generating a list of **N** items, generate a single item **N** times. This avoids the language model getting stuck in a repetitive loop.
- In order to improve output quality, generate many completions and then rank them heuristically.

[Mishra et al.](https://arxiv.org/pdf/2109.07830.pdf) perform an extensive empirical analysis of how to construct effective prompts for GPT3. In particular, they propose a set of reframing techniques for making an arbitrary prompt more likely to generate a successful completion. These techniques include:
- Use low-level patterns from other examples to make a given prompt easier to understand for an LLM.
- Explictly itemize instructions into bulleted lists. Turn negative statements such as *don't create questions which are not* to *create questions which are*.
- When possible break down a top-level task into different sub-tasks that can be executed in parallel or sequentially.
- Avoid repeated and generic statements when trying to solve a very specific task. For example, instead of saying *Answer the following question* for a math problem, say *Calculate answer to the following question. You need to either add or subtract numbers...*

The researchers further demonstrate that these reframed prompts significantly improve performance in few-shot and zero-shot settings, generalize performance across model types, and are even able to outperform (smaller) traditionally supervised models. 
 

## Practical Advice for Prompts

## Automated Prompt Generation

Given the finicky nature of manual prompt engineering, there have been a number of promising research efforts dedicated to developing automated prompting techniques for different LLMs. 

[Shin, Razeghi, and Logan et al.](https://aclanthology.org/2020.emnlp-main.346.pdf) developed a gradient-guided search technique for automatically producing prompts via a set of trigger tokens. When their technique was applied to masked language models (MLM) such as RoBERTa, they were able to elicit impressive performance on tasks such as sentiment analysis, natural language inference, and fact retrieval even outperforming finetuned models in low-data regimes.

[Jiang and Xu et al.](https://watermark.silverchair.com/tacl_a_00324.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAArswggK3BgkqhkiG9w0BBwagggKoMIICpAIBADCCAp0GCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM2owX5YtSdXpjV-uRAgEQgIICbiFIRZ2N5-ZjSsVvVpS-XpBuIN-JYsPqY9abB-KkzX31AQ1Pe1Ou6cJF84R7W1MjuCDc6oGv7Hws7n1YXbovGWgdfyKGZXv4FjlfwLeUo9Fk4BHzLrvanZ_CVtGpC9eFX8FoEnRzVhFlJb3-dYorzbEomACsbhTgSJ_pOWKCArc3Yr_Wf9m-hPpp4BGi90sI-P5wtLtY8haftJYdEDVi9SMi8R5NLj_GWgeFs5JufRhMM9xRwXJu2R4y6Vv7-wP3oQlfFPBlYFO_gdEQek74D_otcYDPlujm9hCUFtOeh018jaRT7thSHx8R-AcIPPntN0vsjDPnijsneZbfB0wwCmDDfxj9A7IzotcyVrcK0_uTvKDZfvA09FcPBlmoUcvQ1Xq_UZeV9FEn8d4Ih2cvPSduaVlRaKrub_7MjC9KPiAoQvEEH13QUf32pWswe_71rPdbLTSd0t2TacecO2OqaCcYF8WkgU3IZ9zoKCsBlFnWnVlQmAgG21PYHivfjLgzKd4r8hhVaAzxOxDbzyWmbC9auHT4DRRe8Pkw1fKl82lRYOw9dDtEPKQRAgXQ-ZB3NpXexOQo7akZ1Zx_2zxylHYErCDzHSEqQnvp4XlE-RKdkgUD95n9fPGE2httBmRzcpTh2rtuV645OPGj7SXayfoQv0CwVdNRRKGiD0RT4dyX5NueIF748z1lg7B4ubFBOqCCd__VMoHTOFuAZwnjR49dUJ0k1KyGHhNmS2ru3JL8cBHpixS5H2JomEDofmVpbIv83mIWgcdroVCT2Hhuvq7cr1BQUrQWvTsKWQhENrCnUHN4LLp43bgoYfkwg8I) proposed using mining and paraphrasing techniques to generate optimal prompts for MLM systems such as BERT, demonstrating a nearly 10% boost in accuracy of relational knowledge extraction. 

[Li et al.](https://arxiv.org/pdf/2101.00190.pdf) proposed an alternative technique that uses a learned, continuous vector (called a *prefix*) that is prepended to the input of generative models whose other parameters are held fixed.
The researchers used prefix tuning for [GPT2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and [BART](https://arxiv.org/abs/1910.13461) generation and was able to outperform finetuned models with 1000x parameters in full-data and low-data settings.


## Survey of Prompting Use-cases

Part of the magic of LLMs is the sheer number of tasks they are able to perform reasonably well using nothing but few and zero-shot prompting techniques. [Some work](https://arxiv.org/pdf/2206.07682.pdf) argues that these emergent abilities only appear in large language models at a certain scale in terms of parameter size. 

Since their rise, LLMs have been applied in more formal academic contexts on everything from knowledge probing, information extraction, questin answering, text classification, natural language inference, [dataset](https://arxiv.org/pdf/2202.04538.pdf) [generation](https://arxiv.org/pdf/2104.07540.pdf), and [much more](https://arxiv.org/pdf/2107.13586.pdf).

For a look at various applications built using LLMs, check out [this admittedly out-dated link](https://gptcrush.com/). Additionally for a neat collection of demonstrations showing prompt-based generation of everything from job application letters to dad jokes, check out [Gwern's article](https://www.gwern.net/GPT-3).

If you want to keep up-to-date on the latest and greatest in prompt engineering tips and tricks, check out [Riley Goodside's work](https://twitter.com/goodside).

## Infrastructure for Prompt Engineering

While prompt engineering is still a relatively nascent concept, it is clear that it requires new interfaces for application development. There have been a number of projects released providing infrastructure for easier prompt design.

[Bach and Sanh et al.](https://arxiv.org/pdf/2202.01279.pdf) built PromptSource, an integrated development environment to systematize and crowdsource best practices for prompt engineering. This includes a templating languaging for defining data-linked prompts and general tools for prompt management.

In related work [Strobelt et al.](https://arxiv.org/pdf/2208.07852.pdf) developed PromptIDE, a handy visual platform to experiment with prompt variations, track prompt performance, and iteratively optimize prompts. I like the general direction of work like this because it suggests that if we systematize the search process for optimal prompts, then one outcome is an AutoML-style solution for prompt engineering.

While much of the work so far in prompting has focused on single-step prompts, we must weave together multiple prompting sequences to get more sophisticated applications. [Wu et al.](https://arxiv.org/pdf/2203.06566.pdf) formalize this in the notion of an LLM chain and propose PromptChainer as a tool to design these multi-step LLM applications. What's powerful about this platform is that it ties together not just prompting steps but also external API calls and user inputs, forming almost a Webflow for prompt engineering. 

## Prompt Engineering Security

One interesting and and concerning phenomenon observed in building LLM applications is the appearance of prompt-based security exploits. More specifically, [various](https://twitter.com/goodside/status/1569128808308957185?s=20&t=U5pyMwrmQ8NADdUIGFLG3A) [people](https://simonwillison.net/2022/Sep/12/prompt-injection/) have noted that by leveraging carefully-crafted inputs, LLMs can generate the "secret" prompts they use in the backend as well as leak credentials or other private information. This has drawn natural comparisons to old-school [SQL injection attacks](https://portswigger.net/web-security/sql-injection).

As of now, there aren't super robust mechanisms to address this issue. Instead people have proposed workarounds using [different formatting of the inputs](https://twitter.com/goodside/status/1569457230537441286) but it is clear more work needs to be done to prevent these exploits.


## Final Thoughts
- Prompt engineering is still a nascent field but a lot of exciting work is being done to understand how far these techniques will take us
- Is it a science? Is it voodoo magic? Hard to say at this point in time, but it's clear that significant energy is being spent to understand how to best become LLM whisperers. 

- Make the comparison to hyperparameter tuning. The evolution of prompt engineering may take a similar form. 


- What will prompt engineering look like once GPT4+ come out
  - importance of carefully tuned prompts may go away
  - we are still a ways away from that outcome 
    - draw analogy to development of improved hyperparameter tuning techniques (autoML)
- A few trends worth pointing out
  - cost of API calls going down
  - push toward more OSS models alla stable diffusion
  - GPT4+

## Further Reading


## Papers
- [Prompt programming for LLMs: Beyond the few-shot paradigm](https://arxiv.org/pdf/2102.07350.pdf)
  - Function of few-shot examples is better described as locating an already learned task rather than meta learning
  - Propose idea of a “metaprompt” that seeds the model to generate its own natural language prompts for a range of tasks
  - Few shot paradigm can be matched or exceeded by simple 0-shot prompts
  - Provide a few good principles for prompt programming
  - "We have observed, for instance, that GPT-3 is much more reliable at noticing when a passage is bizarre or contains errors than it can produce non-bizarre passages without errors"

- [Reframing Instructional prompts in GPTK's Language](https://arxiv.org/pdf/2109.07830.pdf)
  - same reframed instructions boost few-shot performance of GPT3 series by 12.5% and 6.7%
  Reframing prompts into multiple sub-tasks
  - Five diverse framing techniques
  - GPT2/3 models are considered
  - Reframed prompts on gpt3-instruct score roughly 17% higher than gpt2large that is supervised on 1k instances
  - Short prompts that contain concrete statements and avoid terms associated with background knowledge improve GPT3’s response to instructions
  - Principles
    - Use low-level patterns
    - Itemizing instructions
    - Break it down
      - Turn negative statements to positive ones
    - Enforce constraint
      - Add an enforcement statement to improve model output by constraining it to provided types
    - Specialize the instructions
      - Drop all repeated and generic statements

- [PromptSource](https://github.com/bigscience-workshop/promptsource)
  - collection of prompts across datasets
  - 2000 English prompts across 170+ english datasets (Public Pool of Prompts)
    - Can adapt these for a given task
  - (1) a templating language for defining data-linked prompts
  - (2) an interface that lets users quickly iterate on prompt development by observing outputs of their prompts on many examples
  - (3) community driven set of guidelines for contributing new prompts
  - Global view for browsing datasets and existing prompt templates
  - Local view facilitates iteration on prompt wording and metadata
  - Prompts are more functions than labels
    - More like programming than typical data annotation
  - Dataset-level choices
  - How to inspect and debug across many examples

- [AI Chains](https://arxiv.org/pdf/2110.01691.pdf)
  - Define a set of LLM primitive operations useful for chain construction
  - Unit testing sub-components of a Chain
  - LLM challenges
    - Lack multi-step reasoning capabilities
    - They grasp the form, not the meaning of language
    - Suffer from exposure bias
    - Sensitive to input prompts
  - derivative work: [Promptchainer](https://arxiv.org/pdf/2203.06566.pdf)

- [Making Pretrained Language Models Better Few Shot Learners](https://aclanthology.org/2021.acl-long.295.pdf)
  - Better few shot finetuning of language models
  - Simple and complementary techniques for finetuning language models on a small number of annotated examples
  - 1) prompt-based fine-tuning together with a novel pipeline for automating prompt generation
  - 2) a refined strategy for dynamically and selectively incorporating demonstrations into each context
  - Up to 30% absolute improvement and 11% on average across all tasks
- [Autoprompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts](https://aclanthology.org/2020.emnlp-main.346.pdf)
  - Autoprompt
    - Automated method to create prompts for a diverse set of tasks based on gradient-guided search
    - Prompts elicit more accurate factual knowledge from MLMs than manually created prompts
    - Used as relation extractors more effectively than supervised relation extraction models
    - Create a prompt by combining the original task inputs with a collection of trigger tokens according to a template (same set of trigger tokens used for all inputs)
  - Non finetuned MLM performs well on both tasks achieving 91% accuracy on SST-2
  - Combined prompt + trigger words + [MASK] token
    - Set of label tokens that correspond to particular label y
  - Trigger tokens
    - Initialized to [MASK] and then iteratively updated to maximize label likelihood
    - Compute first order approx. of change in log likelihood that would be produced by swapping the jth trigger token with another token
    - Then identify candidate set V_cand of top k tokens estimated to cause greatest increase
- [Fantastically Ordered Prompts](https://arxiv.org/pdf/2104.08786.pdf)
  - Order of examples matters in few shot LLM priming
  - Construct artificial development set to identify performant prompts
  - Sample order makes as much difference as right template
  - Increasing model size somewhat addresses order sensitivity
  - Increasing # of training samples tends to help in terms of performance but still a high variance remains
  - Performant prompts are not transferable across models
  - Select prompt orders to achieve better performance
  - Generative process
    - Objective is to generate a probig set that shares a similar distribution to training samples
    - Identify best ordering via one of two techniques
    - Global entropy
    - Local entropy
    - Entropy based probing is effective across templates
    - Sentence pair tasks remain challenging for smaller-sized models even with performant permutation selection
    - Perhaps no good prompt exists
- [Interactive and Visual Prompt Engineering for Ad-hoc Task Adaptation](https://arxiv.org/pdf/2208.07852.pdf)
  - PromptIDE
  - Allows users to experiment with prompt variations
  - Visualize prompt performance
  - Iteratively optimize prompts
  - Problem of prompt engineering is still highly dependent on specific wording choices for templates, reflected in a high variance in accuracy
  - Goals
  - Support a broad set of ad-hoc NLP tasks
  - Faster and more informed prompt writing through feedback from data
  - Ground prompt choices in quantitative measures
  - Ease deployment of models to end uses
  - Tasks
  - Formulating and trying out prompts and prompt variations
  - Encoding prediction details of the model
  - Testing promising prompts on task performance
  - Export prompt for concrete deployment

- [Pretrain, prompt, and predict: A systematic survey of prompting methods in NLP](https://arxiv.org/pdf/2107.13586.pdf)
  - two types of prompts
    - cloze prompts
    - discrete prompts
      - prompt mining
      - prompt paraphgrasing
      - gradient-based search
      - prompt generation
    - continuous promps
      - 
    - prefix prompts
- [Calibrate before use: improving few shot performance of language models](https://arxiv.org/pdf/2102.09690.pdf)
  - changing a prompt immediately leads to a new model
  - gpt3's accuracy is highly dependent on selection and permutation of training examples
  - variance persists with more data and larger models
  - majority label bias
  - recency bias
  - common token bias by predicting tokens that are in its pretraining distribution

## Miscellaneous
- https://gptcrush.com/
  - list of GPT apps
- GPT3 use-cases
  - Counting correctly by enumerating steps    
    - https://twitter.com/goodside/status/1564320893194608640?s=20&t=b80W2bAXo1GpEj3GXFlKnw
  - Identify whether a poem is a haiku
    - https://twitter.com/goodside/status/1564379216271081473?s=20&t=b80W2bAXo1GpEj3GXFlKnw
  - Make plotting library
    - https://twitter.com/goodside/status/1564051177246973953?s=20&t=b80W2bAXo1GpEj3GXFlKnw
  - Generating a number of distinct structured files 
    - https://twitter.com/goodside/status/1563989550808154113?s=20&t=b80W2bAXo1GpEj3GXFlKnw
  - How to format inputs
    - https://twitter.com/goodside/status/1563717590257750016?s=20&t=b80W2bAXo1GpEj3GXFlKnw
