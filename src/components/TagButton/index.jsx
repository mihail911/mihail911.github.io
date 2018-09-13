import React from 'react';
import Link from 'gatsby-link';
import './style.scss';

class TagButton extends React.Component {
  render() {
    return (
      <span className="buttonWrapper">
        <Link to={this.props.tagSlug} className="post__meta-category-link">
          {this.props.tag}
        </Link>
      </span>
    );
  }
}

export default TagButton;
