import React from 'react';
import nameToPic from '../../imgs';
import './style.scss';

class ProjectDescription extends React.Component {
  // TODO (mihail): Figure out how to increase width of entry
  render() {
    let containerClassName = 'container';
    // Set a specific styling element for container div if it is at top or bottom or screen, namely
    // adding a bottom border
    if (this.props.edge) {
      containerClassName = 'container-edge';
    }

    return (
      <div className={containerClassName}>
        <div className="image">
          <a href={this.props.link}>
            <img src={nameToPic[this.props.picName]} alt={this.props.title} />
          </a>
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
