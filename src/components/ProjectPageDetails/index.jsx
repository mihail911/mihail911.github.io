import React from 'react';
import './style.scss';
import ProjectDescription from '../ProjectDescription';


class ProjectPageDetails extends React.Component {
  render() {
    return (
      <div>
        <div className="section-title">
          Engineering Playground
        </div>
        <ProjectDescription
          title="Rick and Mortify"
          picName="randm"
          link="https://rickandmortify.com/"
          description="When will we see an AI system listed in the closing credits of your favorite feature-length film? Rick and Mortify is a tool we built using the state-of-the-art in large vision and language models to create never-before-seen episodes of Rick and Morty with minimal human intervention."
        />
        <ProjectDescription
          title="AutoPlot"
          picName="autoplot"
          link="https://autoplot.app"
          description="What if you could do any data analysis by just speaking in natural language? AutoPlot is a tool we developed to explore the limits of next generation human interfaces for visually understanding data."
        />
        <ProjectDescription
          title="Confetti AI"
          picName="confetti"
          link="https://www.confetti.ai/"
          description="Confetti AI is an educational platform helping people learn the skills to succeed in artificial intelligence careers. It provides a collection of targeted resources and tools to empower the next generation of AI practitioners. We were acquired by Towards AI in 2022."
        />
        <ProjectDescription
          title="Where Does Tax Money Go"
          picName="taxes"
          link="https://wheredoestaxmoneygo.mihaileric.com/"
          description="Every year, your hard-earned money gets shuffled to taxes. The purpose of this little project was an attempt to answer the simple question: what does your tax money actually pay for?"
        />
        <ProjectDescription
          title="Artificial Intelligence School"
          picName="aiSchool"
          link="https://itunes.apple.com/us/app/artificial-intelligence-school/id1369987569?mt=8"
          description="AI School is a mobile app designed to teach the basics of artificial intelligence, machine learning, and deep learning. Includes a comprehensive lesson plan for learning fundamental principles."
          meta="Available on both the App Store and Google Play!"
          edge="true"
        />

        {/* PUBLICATIONS */}
        <div className="section-title">
          Publications
        </div>
        <ProjectDescription
          title="A Machine Learning Primer"
          picName="primer"
          link="https://www.confetti.ai/assets/ml-primer/ml_primer.pdf"
          authors="Mihail Eric"
          description="I provide an overview of key machine learning concepts that show up in data science and ML interviews. This primer includes core theory as well as practice problems to test your knowledge."
          meta="Confetti AI"
        />
        <ProjectDescription
          title="Example-Driven Intent Prediction With Observers"
          picName="observers"
          link="https://arxiv.org/abs/2010.08684"
          authors="Shikib Mehri, Mihail Eric, Dilek Hakkani-Tur"
          description="We propose two approaches for improving the generalizability of utterance classification models (example-driven training and observers) that when used in combination achieve state-of-the-art results in full-data and few-shot settings on several intent prediction datasets."
          meta="arXiv:2010.08684"
        />
        <ProjectDescription
          title="DialoGLUE: A Natural Language Understanding Benchmark for Task-Oriented Dialogue"
          picName="dialoglue"
          link="https://arxiv.org/abs/2009.13570"
          authors="Shikib Mehri, Mihail Eric, Dilek Hakkani-Tur"
          description="We introduce DialoGLUE (Dialogue Language Understanding Evaluation), a public benchmark consisting of 7 task-oriented dialogue datasets covering 4 distinct natural language understanding tasks. We introduce new models that achieve state-of-the-art results on 5/7 datasets in the benchmark."
          meta="arXiv:2009.1357"
        />
        <ProjectDescription
          title="Proceedings of 2nd ACL NLP for Conversational AI Workshop"
          picName="nlp4convai"
          link="https://www.aclweb.org/anthology/2020.nlp4convai-1.0.pdf"
          authors="Tsung-Hsien Wen, Asli Celikyilmaz, Zhou Yu, Alexandros Papangelis, Mihail Eric, Anuj Kumar, Inigo Casanueva, Rushin Shah"
          description="The goal of this workshop is to bring together  NLP researchers and practitioners in different fields,
          alongside experts in speech and machine learning, to discuss the current state-of-the-art and new
          approaches, to share insights and challenges, to bridge the gap between academic research and realworld product deployment, and to shed the light on future directions."
          meta="ACL 2020"
        />
        <ProjectDescription
          title="Beyond Domain APIs: Task-oriented Conversational Modeling with Unstructured Knowledge Access"
          picName="kmdm"
          link="https://arxiv.org/abs/2006.03533"
          authors="Seokhwan Kim, Mihail Eric, Karthik Gopalakrishnan, Behnam Hedayatnia, Yang Liu, Dilek Hakkani-Tur"
          description="We propose to expand coverage of task-oriented dialogue systems by incorporating external unstructured knowledge sources."
          meta="SIGDial 2020, arXiv:2006.03533"
        />
        <ProjectDescription
          title="Policy-Driven Neural Response Generation for Knowledge-Grounded Dialogue Systems"
          picName="policy"
          link="https://arxiv.org/abs/2005.12529"
          authors="Behnam Hedayatnia, Karthik Gopalakrishnan, Seokhwan Kim, Yang Liu, Mihail Eric, Dilek Hakkani-Tur"
          description="We propose a technique for using a dialogue policy to plan the content and style of target responses in the form of an action plan."
          meta="Preprint 2020, arXiv:2005.12529"
        />
        <ProjectDescription
          title="Just Ask: An Interactive Learning Framework for Vision and Language Navigation"
          picName="vdn"
          link="https://arxiv.org/abs/1912.00915"
          authors="Ta-Chung Chi, Mihail Eric, Seokhwan Kim, Minmin Shen, Dilek Hakkani-Tur"
          description="We propose a novel scheme for interactive human-in-the-loop learning, achieving more data-efficient performance on a vision and language task."
          meta="AAAI 2020, arXiv:1912.00915"
        />
        <ProjectDescription
          title="MultiWOZ 2.1: Multi-Domain Dialogue State Corrections and State Tracking Baselines"
          picName="dst"
          link="https://arxiv.org/abs/1907.01669"
          authors="Mihail Eric*, Rahul Goel*, Shachi Paul, Abhishek Sethi,
            Sanchit Agarwal, Shuyang Gao, Dilek Hakkani-Tur"
          description="We release an updated version of the Cambridge MultiWOZ dataset with dialogue state annotation corrections and corresponding state tracking baselines."
          meta="Preprint 2019, arXiv:1907.01669"
        />
        <ProjectDescription
          title="Key Value Retrieval Networks for Task-Oriented Dialogue"
          picName="kv"
          link="https://arxiv.org/abs/1705.05414"
          authors="Mihail Eric, Lakshmi Krishnan, Francois Charette,
            Christopher D. Manning"
          description="We demonstrate the efficacy of a new neural dialogue agent that is able to effectively sustain grounded, multi-domain discourse through a novel key-value retrieval mechanism."
          meta="SIGDial 2017 Oral Presentation, arXiv:1705.05414"
        />
        <ProjectDescription
          title="The Pragmatics of Indirect Commands in Collaborative Discourse"
          picName="cards"
          link="https://arxiv.org/abs/1705.03454"
          authors="Matthew Lamm* and Mihail Eric*"
          description="We show that models with domain-specific grounding can effectively realize the pragmatic reasoning that is necessary for more robust natural language interaction."
          meta="International Conference on Computational Semantics 2017, arXiv:1705.03454"
        />
        <ProjectDescription
          title="Learning Symmetric Collaborative Dialogue Agents with Dynamic Knowledge Graph Embeddings"
          picName="mutual"
          link="https://arxiv.org/abs/1704.07130"
          authors="He He, Anusha Balakrishnan, Mihail Eric, Percy Liang"
          description="To model both structured knowledge and unstructured language in a novel dialogue setting, we propose a neural model with dynamic knowledge graph embeddings that evolve as the dialogue progresses."
          meta="ACL 2017, arXiv:1704.07130"
        />
        <ProjectDescription
          title="A Copy-Augmented Sequence-to-Sequence Architecture Gives Good Performance on Task-Oriented Dialogue"
          picName="copy"
          link="https://arxiv.org/abs/1701.04024"
          authors="Mihail Eric, Christopher D. Manning"
          description="We show the effectiveness of simple sequence-to-sequence neural architectures with a copy mechanism, outperforming more sophisticated models on a standard task-oriented dialogue dataset."
          meta="EACL 2017 Oral Presentation, arXiv:1701.04024"
        />
        <ProjectDescription
          title="SceneSeer: 3D Scene Design with Natural Language"
          picName="sceneseer"
          link="https://arxiv.org/abs/1703.00050"
          authors="Angel X. Chang, Mihail Eric, Manolis Savva, Christopher D. Manning"
          description="We present SceneSeer: an interactive text to 3D scene generation system with a learned spatial knowledge base that allows a user to design 3D scenes using natural language."
          meta="Preprint 2017, arXiv:1703.00050"
          edge="true"
        />

        {/* TECHNICAL REPORTS */}
        <div className="section-title">
          Technical Reports
        </div>
        <ProjectDescription
          title="Using Contextual Information for Neural Natural Language Inference"
          picName="nnli"
          link="https://cs224d.stanford.edu/reports/billovits.pdf"
          authors="Chris Billovits* and Mihail Eric*"
          description="We investigate neural memory network architectures for the task of natural language inference and propose models for using attention across relevant semantic phrases to inform common sense reasoning."
          meta="Preprint 2016"
        />
        <ProjectDescription
          title="Hitting Depth: Investigating Robustness to Adversarial Examples in Deep Convolutional Neural Networks"
          picName="adversarial"
          link="http://cs231n.stanford.edu/reports/2016/pdfs/119_Report.pdf"
          authors="Chris Billovits* and Mihail Eric* and Nipun Agrawal*"
          description="We show a process for visualizing and identifying changes in activations between adversarial images and their regular counterparts and propose a Bayesian framework for improving prediction accuracy on adversarial examples."
          meta="Preprint 2016"
        />
        <ProjectDescription
          title="Wordwise Inference and Entailment Now"
          picName="wien"
          link="https://www.semanticscholar.org/paper/WIEN-%3A-Wordwise-Inference-and-Entailment-Now-Or-%3A-Billovits-Eric/b525f726265aca03d07021b0e9dec14c6ace0890"
          authors="Chris Billovits* and Mihail Eric* and Chris Guthrie*"
          description="We implement a random forest classifier with a carefully engineered and selected collection of linguistic and semantic features for the task of natural language inference, achieving an F1 of 80.9% on the SemEval-2014 Dataset."
          meta="Preprint 2016"
          edge="true"
        />
      </div>
    );
  }
}

export default ProjectPageDetails;
