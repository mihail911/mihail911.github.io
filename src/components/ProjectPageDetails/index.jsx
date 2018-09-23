import React from 'react';
import './style.scss';
import ProjectDescription from '../ProjectDescription'



class ProjectPageDetails extends React.Component {
  render() {
    return (
      <div>
        {/* ENGINEERING PLAYGROUND */}
        <div className="section-title">
          Engineering Playground
        </div>
        <ProjectDescription
          title="Artificial Intelligence School"
          picName="aiSchool"
          link="https://itunes.apple.com/us/app/artificial-intelligence-school/id1369987569?mt=8"
          authors="Mihail Eric"
          description="AI School is a mobile app designed to teach the basics of artificial intelligence, machine learning, and deep learning. Includes a comprehensive lesson plan to teach the fundamental principles."
          meta="Available on both the App Store and Google Play!"
          edge="true"
        />

        {/* PUBLICATIONS */}
        <div className="section-title">
          Publications
        </div>
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
          description="We demonstrate the efficacy of a new neural dialogue agent that is able to effectively sustain grounded, multi-domain discourse through a novel key-value retrieval mechanism."
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
          authors="Chris Billovits* and Mihail Eric* and Chris Guthrie"
          description="We implement a random forest classifier with a carefully engineered and selected collection of linguistic and semantic features for the task of natural language inference, achieving an F1 of 80.9% on the SemEval-2014 Dataset."
          meta="Preprint 2016"
          edge="true"
        />
      </div>
    );
  }
}

export default ProjectPageDetails;
