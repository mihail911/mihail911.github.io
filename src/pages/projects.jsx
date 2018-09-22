import React from 'react';
import Helmet from 'react-helmet';
import Sidebar from '../components/Sidebar';
import ProjectPageDetails from '../components/ProjectPageDetails'

class ProjectsRoute extends React.Component {

  render() {
    const subTitle = 'description of Mihail Eric\'s projects';
    return (
      <div>
        <Helmet>
          <title>Projects</title>
          <meta name="description" content={subTitle} />
        </Helmet>
        <Sidebar {...this.props} />
        <div className="content">
          <div className="content__inner">
            <ProjectPageDetails />
          </div>
        </div>
      </div>
    );
  }
}

export default ProjectsRoute;

export const pageQuery = graphql`
  query ProjectQuery {
    site {
      siteMetadata {
        title
        subtitle
        copyright
        menu {
          label
          path
        }
        author {
          name
          email
          twitter
          github
          linkedin
        }
      }
    }
  }
`;
