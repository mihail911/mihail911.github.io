import React from 'react';
import './style.scss';
import '../../assets/fonts/fontello-771c82e0/css/fontello.css';
import '../../assets/fonts/fontello-b224adf3/css/fontello.css';
import '../../assets/fonts/fontello-19db5942/css/fontello.css';

class Links extends React.Component {
  render() {
    const author = this.props.data;
    const links = {
      twitter: author.twitter,
      github: author.github,
      email: author.email,
      linkedin: author.linkedin
    };

    return (
      <div className="links">
        <ul className="links__list">
          <li className="links__list-item">
            <a href={`https://www.twitter.com/${links.twitter}`} target="_blank" >
              <i className="icon-twitter" />
            </a>
          </li>
          {/* TODO (mihail): Figure out how to make background for icons red when hovered */}
          <li className="links__list-item">
            <a href={`https://www.github.com/${links.github}`} target="_blank" >
              <i className="icon-github" />
            </a>
          </li>
          <li className="links__list-item">
            <a href="./mihaileric.resume.pdf">
              <i className="icon-newspaper" />
            </a>
          </li>
          <li className="links__list-item">
            <a href={`https://www.linkedin.com/in/${links.linkedin}`}>
              <i className="icon-linkedin" />
            </a>
          </li>
          <li className="links__list-item">
            <a href="https://scholar.google.com/citations?user=4kJYLpcAAAAJ&hl=en">
              <i className="icon-graduation-cap" />
            </a>
          </li>
        </ul>
      </div>
    );
  }
}

export default Links;
