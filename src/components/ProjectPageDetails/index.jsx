import React from 'react';
import './style.scss';
import ProjectDescription from '../ProjectDescription'



class ProjectPageDetails extends React.Component {
  render() {
    return (
      <div>
        <div className="section-title">
          Engineer{'\''}s Playground
        </div>
        <ProjectDescription
          title="Key Value Retrieval Networks for Task-Oriented Dialogue"
          link="https://arxiv.org/abs/1705.05414"
          authors="Mihail Eric, Lakshmi Krishnan, Francois Charette,
            Christopher D. Manning"
          description="We demonstrate the efficacy of a new neural dialogue agent that is able to effectively sustain grounded, multi-domain discourse through a novel key-value retrieval mechanism."
          meta="SIGDial 2017 Oral Presentation, arXiv:1705.05414"
        />
        <div className="section-title">
          Publications
        </div>
        <div className="section-title">
          Technical Reports
        </div>
      </div>
    );
  }
}

export default ProjectPageDetails;
