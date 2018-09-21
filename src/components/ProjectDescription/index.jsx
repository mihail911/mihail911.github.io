import React from 'react';
import biblePic from '../../pages/pages/imgs/42-line-bible.jpg';
import './style.scss';
import Link from 'gatsby-link'


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
            <a href={this.props.link}>
              {this.props.title}
            </a>
          </div>
          <div className="description">
            {this.props.description}
          </div>
          <div className="authors">
            {this.props.authors}
          </div>
          <div className="meta">
            {this.props.meta}
          </div>
        </div>
      </div>
    );
  }
}

export default ProjectDescription;
