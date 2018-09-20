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
        <ProjectDescription />
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
