import React from 'react';
import biblePic from '../../pages/42-line-bible.jpg';
import './style.scss';


class ProjectDescription extends React.Component {
  // TODO (mihail): Figure out how to increase width of entry
  render() {
    return (
      <div className="container">
        <div className="image">
          <img src={biblePic} alt="alt text" />
        </div>
        <div className="project-details">
          <div className="title">
            Title
          </div>
          <div className="authors">
            Authors
          </div>
          <div className="description">
            Description
          </div>
          <div className="meta">
            Meta
          </div>
        </div>
      </div>
    );
  }
}

export default ProjectDescription;
