---
title: "Deploying a State-of-the-Art Question Answering System With 60 Lines of Python Using HuggingFace and Streamlit"
date: "2020-09-19T23:40:37.121"
layout: post
draft: false
path: "/posts/state-of-the-art-question-answering-streamlit-huggingface/"
tags:
  - "Natural Language Processing"
  - "Deep Learning"
description: "I show how you can deploy a performant question-answering system with just 60 lines of Python. "
---

<figure>
    <img src="./blastoff.jpg" alt="blastoff spaceship">
</figure>

Nowadays, the machine learning and data science job landscape is changing rapidly.

Within industry, the skills that are becoming most valuable aren't knowing how to tune a ResNet on an image dataset. In fact, the prevalence of well-designed frameworks such as PyTorch and Tensorflow are making these skills increasingly easy to pickup.

Rather as many larger enterprises look to adopt machine learning as part of their business offerings, the skills that are in high demand are knowing how to solve the "last mile problem." In other words, how do you go from a trained, functional model sitting on your local machine to a deployed service that can be used by customers, typically via a web API?

Solving this problem is less about the right hyperparameters or features needed to eke out the last percentage point on a task but more about knowing how to engineer a deployment pipeline.

This means that engineering and infrastructure requirements are emerging as the biggest bottlenecks in deploying real world machine learning systems. The statistics are sobering: [87% of data science project never make it into production](https://venturebeat.com/2019/07/19/why-do-87-of-data-science-projects-never-make-it-into-production/).

Thankfully, we are also seeing the emergence of powerful libraries that help address this last mile problem. One library in particular, called Streamlit, is a powerful player in this space that offers a low-effort solution to the deployment.

In this post, we will show how with [Streamlit](https://www.streamlit.io/) and the [HuggingFace Transformers library](https://huggingface.co/transformers/) we need only **60 lines of Python** to deploy an interactive web app making calls to a state-of-the-art neural question answering system that can query all of Wikipedia.

Let's get started!

## Use-Case

Our app will use a powerful neural model can be used to answer questions about any arbitrary Wikipedia article. It will allow users to retrieve any Wikipedia article and then ask the model to read and extract bits of information from it.

## Model

The question answering model used is a variant of [DistilBert](https://arxiv.org/pdf/1910.01108.pdf), a neural Transformer model with roughly 66 million parameters. Using a smaller model ensures you can still run inference in a reasonable time on commodity servers.

## Code

We first load up our question answering model via a pipeline:

```python

from typing import Dict

import streamlit as st
import wikipedia
from transformers import Pipeline
from transformers import pipeline

NUM_SENT = 10

@st.cache
def get_qa_pipeline() -> Pipeline:
    qa = pipeline("question-answering")
    return qa


def answer_question(pipeline: Pipeline, question: str, paragraph: str) -> Dict:
    input = {
        "question": question,
        "context": paragraph
    }
    return pipeline(input)
```

Here we are using a pipeline object that wraps around a pretrained model from the  [Transformers library](https://github.com/huggingface/transformers). Note we are using the *@st.cache* Streamlit decorator which prevents unnecessary reloads of the model, since this can be computationally expensive.

Next we provide functionality for getting articles from Wikipedia:

```python

@st.cache
def get_wiki_paragraph(query: str) -> str:
    results = wikipedia.search(query)
    try:
        summary = wikipedia.summary(results[0], sentences=NUM_SENT)
    except wikipedia.DisambiguationError as e:
        ambiguous_terms = e.options
        return wikipedia.summary(ambiguous_terms[0], sentences=NUM_SENT)
    return summary


def format_text(paragraph: str, start_idx: int, end_idx: int) -> str:
    return paragraph[:start_idx] + "**" + paragraph[start_idx:end_idx] + "**" + paragraph[end_idx:]
```

This uses the provided query to make a call to the [Python wikipedia library](https://wikipedia.readthedocs.io/en/latest/quickstart.html#quickstart). The second function will be used once our model returns a value to highlight the answer within the paragraph.

Finally, we provide the main engine of the app, which renders the text inputs using Streamlit and makes the subsequent calls to the above functions:

```python

if __name__ == "__main__":
    """
    # Wikipedia Article
    """
    paragraph_slot = st.empty()
    wiki_query = st.text_input("WIKIPEDIA SEARCH TERM", "")
    question = st.text_input("QUESTION", "")

    if wiki_query:
        wiki_para = get_wiki_paragraph(wiki_query)
        paragraph_slot.markdown(wiki_para)
        # Execute question against paragraph
        if question != "":
            pipeline = get_qa_pipeline()
            st.write(pipeline.model)
            st.write(pipeline.model.config)
            try:
                answer = answer_question(pipeline, question, wiki_para)

                start_idx = answer["start"]
                end_idx = answer["end"]
                paragraph_slot.markdown(format_text(wiki_para, start_idx, end_idx))
            except:
                st.write("You must provide a valid wikipedia paragraph")

```

All we have to do to deploy the app locally is save the code within a file **app.py** and run the following from the command line:

```python
streamlit run app.py
```

And with that, we have a functional state-of-the-art question-answering system deployed as a web application!

There's a lot more that can be done with Streamlit, so I encourage you to check out the [documentation](https://docs.streamlit.io/en/stable/api.html). You can check out all the code for this application [here](https://github.com/mihail911/question-answering-with-streamlit).