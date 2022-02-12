---
title: "MLOps is a Clusterf*ck, But That's to be Expected"
date: "2021-01-10T23:40:37.121"
layout: post
draft: true
path: "/posts/mlops-is-a-clusterf*ck/"
tags:
  - "Data Science"
  - "Machine Learning"
  - "A.I."
  - "Engineering"
description: ""
---

<figure>
    <img src="./nasa-Q1p7bh3SHj8-unsplash.jpg" alt="world in data">
</figure>

MLOps is a Shitshow, but that's to be expected

Does this sound familiar? You read an article that said machine learning was *the* job to get in 2022, being not only super in-demand but commanding among the [highest industry salaries around](https://artificialintelligence-news.com/2019/03/15/machine-learning-jobs-high-paying-demand/). That sounds nice: job security and money. What's not to like?

You decide you're going to go for it, learn the skills to be a machine learning engineer, do a few side projects to beef up your resume, and land that job. You're feeling good. I mean how hard could it possibly be?

You remember seeing on Twitter that there's some [course at Berkeley for full-stack deep learning](https://fullstackdeeplearning.com/) that's supposed to be really good. You do a few lessons and then see this diagram with the tooling required for the modern ML ecosystem:

<figure>
    <img src="./ml_tools_new.png" alt="machine learning tooling diagram from full stack deep learning course">
</figure>

Oof. That's a lot of stuff to learn. You've used some of this tech but what's Airflow? dBt? Weights & Biases? Streamlit? Ray? 

You feel a bit discouraged. So after a hard-day of reviewing course materials you decide you need some inspirational pick-me-ups. Venture capitalists are always good at thinking big, painting the promised land, getting people excited. 

You remember that one VC [Matt Turck](https://mattturck.com/) always does some annual review of what's hot in AI today. Snazzy new tech, that always gets you more pumped than a Boston Dynamics demo video. So you check out [his 2021 review](https://mattturck.com/data2021/) talking about the ML and data landscape. 

This is the first image you see:

<figure>
    <img src="./ml_and_data_landscape.png" alt="machine learning and data landscape ">
</figure>

What. The. Actual. Hell. 

You close your browser, pour yourself a glass of Scotch, and ponder the fickleness of life. Fin.

Today, the machine learning continues to be one of the most talked about and touted technology waves in society promising to revolutionize every corner of society. 

And yet the ecosystem is in a frenzied state. New fundamental science advances come out of every week. Startups and enterprises spray new developer tools into the market trying to capture a chunk of what many speculate to be a market worth between [$40-120 billion by 2025](https://searchsoftwarequality.techtarget.com/feature/Analysts-mixed-on-future-growth-of-MLOps-AutoML-tools). 

Things are moving fast and furious. And yet if you're just entering the discourse, how do you make sense of it all?

In this post, I want to focus the discussion about the state of machine learning operations (MLOps for short) today, where we are, where we are going. 

As a practitioner who's worked at leading AI organizations and also runs a [machine learning consultancy helping companies deliver data-driven value](https://www.pametandata.com/), I've experienced first-hand the trials and tribulations of bringing ML to the real world. I truly believe there's a lot to be optimistic about with machine learning, but the road is not without a handful of speed-bumps. 

Because Google Analytics tells me that 90% of readers are going to drop off after this intro, the TLDR of the post for those that want to be able to tell their colleagues "I got the gist of what he was saying."

**TLDR**: TODO!

Let's begin. 

# What's in a Name

Let's first start with some definitions. MLOps refers to the set of practices and tools to deploy and reliably maintain machine learning systems in production. In short, MLOps is the medium by which machine learning enters and exists in the real world. 

It's a multidisciplinary field that exists at the intersection of devops, data science, and software engineering. 

<figure>
    <img src="./mlops_venn_diagram.png" alt="mlops, data science, software engineering, devops venn diagram">
</figure>

While there continue to be exciting new advances in AI research, today we are in the deployment phase of machine learning. As a consequence, we are seeing an overabundance of new tools being created to standardize and capture companies' workflows delivering machine learning value. 

In the Gartner technology hype cycle paradigm, we are gradually entering the Slope of Enlightenment where we've passed the AGI fear-mongering and Spike Jonze *Her* promises and organizations are now asking the serious operational questions about how they can get the best best bang for their machine learning buck.

<figure>
    <img src="./hype_cycle_ml.png" alt="hype cycle for machine learning ops">
</figure>


# The State of Affairs Today

MLOps is in a wild state today with the tooling landscape offering more rare breeds than an Amazonian rainforest. As an example, most practitioners would agree that monitoring your machine learning models in production is a crucial part of maintaining a robust, performant architecture. However when you get around to picking a provider I can name 6 different options without even trying: Fiddler, Arize, Evidently, Whylabs, Sagemaker Monitoring, Arthur, etc etc. And we haven't even mentioned the pure data monitoring tools. 

Don't get me wrong: it's nice to have options but are these different monitoring tools really *so* differentiated that we need 6+ of them? Is the market for monitoring really *so* big that these are all billion dollar companies?

At least with monitoring, there's generally agreement about what exact part of the machine learning life cycle these companies are trying to own. Other parts of the stack are not as crisply understood and accepted.  

To illustrate this point, it's become popular among companies to make every new tool they build for the MLOps stack some kind of a *store*. We started with [model stores](https://neptune.ai/blog/mlops-model-stores). Then [feature stores](https://www.tecton.ai/blog/what-is-a-feature-store/) emerged on the scene. Now we also have [metric stores](https://medium.com/airbnb-engineering/how-airbnb-achieved-metric-consistency-at-scale-f23cc53dea70). Oh also [evaluation stores](https://gantry.io/). 

My general take is that the machine learning community is particularly creative when it comes to making synonyms for *database*. A more serious take (when you disregard the effect of companies' marketing attempts) is that the entire field is still standardizing the best way to architect fully-fledged ML pipelines. Achieving consensus around best practices will be a 5-10+year transformation easily. 

During a particularly intriguing discussion among practitioners within the [MLOps community](https://mlops.community/), [Lina Weichbrodt](Lina Weichbrodt) made a claim that the *ML stack* is about as general as the *backend programming development stack.* 

There's something particularly astute about that observation, the idea that a canonical *ML stack* is still not well-defined. In that light, when we consider the phases of an MLOps pipeline rather than a clear architecture diagram like this

<figure>
    <img src="./hype_cycle_ml.png" alt="slide about ML ops stack from sculley paper">
</figure>

what we have today is probably something more like this

<figure>
    <img src="./hype_cycle_ml.png" alt="slide about ML ops stack from sculley paper but looking like an amoeba">
</figure>

We have a sense for what a lot of the right pieces are, but the true separation of concerns is still evolving. Hence MLOps tooling companies tend to enter the market addressing a certain niche and then inevitably start to expand amoeba-style into surrounding architectural responsibilities.

I believe that the state of the tooling landscape with constantly changing responsibility shifts and new lines in the sand is especially hardest for newcomers to the field. It's a pretty rough time to be taking your first steps into MLOps. 

I liken MLOps today to the state of modern web development where new tools are coming on the market all the time and there are about 300 different combinations of frameworks that you can use to build a simple "Hello World" webapp. 

In these situations, my recommendation for newcomers is to 1) engage more experienced individuals to help you consider the options, think through different technology, and be a sounding board for "dumb" questions and 2) spend a lot of time building real systems so that you can experience first-hand the painpoints that you need different tools to address. 

And recognize that no one has all the answers. We're all still figuring out the *right* way to do things.

One other thing to appreciate is that it's easy to get the impression that machine learning sophistication among enterprises is incredibly advanced. Based on my experience and those of other practitioners I've spoken with, the reality of ML maturation among enterprises is far more demure than we would be led to believe based on the tooling and funding landscape. 

The truth is there are only a handful of super sophisticated AI-first enterprises with robust machine learning infrastructure in place to handle their petabytes of data. While most companies don't have that scale of data and hence those types of ML requirements, these AI-first enterprises end up defining the narrative of tooling and standards.

In reality, there's a huge long-tail distribution of awesome companies that are still figuring out their ML strategy.

<figure>
    <img src="./ml_sophistication_plot.png" alt="plot of machine learning sophistication among companies with huge long tail">
</figure>

These "ML at reasonable scale" companies (to use [Jacopo's terminology](http://www.jacopotagliabue.it/)), are fantastic businesses in their own right (in diverse verticals like automation, fashion, etc.) with good-sized proprietary datasets (hundreds of gigabytes to terabytes) that are still early in their ML adoption. 

These companies still stand to get their "first-wins" with ML and generally have pretty low-hanging fruit to get those wins. They don't even necessarily require these super advanced sub-millisecond latency hyper-real-time pieces of infrastructure to start levelling up their machine learning. I believe that one of the big challenges for MLOps over the next 10 years will be helping to onboard these classes of businesses.


# What MLOps Can Learn From DevOps Over the Years
- our abstractions are just not that clean yet, leaky in that many aspects of infrastructure trickle into data concerns
- mlops is a mess because there are leaky abstractions across the architecture boundaries, things being redefined
- We can look to past trends in tooling alla devops of 10+ years ago to see how things evolved
- This is a transformation that will be 6-10+ years in the making easily

# Strong Predictions, Weakly Held
- Talk about things to look forward to - trends of the future
  - closing the loop on machine learning systems (monitoring)
  - declarative systems for ML, i do think analogy to DBs is an apt one
- talent shortage is still rampant
- Machine learning tapering Google trend, interest is slowing down?
  - Deployment phase of cycle




- data engineering as an artifact position of the transition








Meanwhile experienced practitioners are inundated with new offerings everyday like a lucky [Bachelor](https://abc.com/shows/the-bachelor) contestant. 


- unless you are a startup founder, many ML practitioners tend to be bearish on the current tooling landscape
