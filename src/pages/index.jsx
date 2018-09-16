import React from 'react';
import Helmet from 'react-helmet';
import PageTemplateDetails from '../components/PageTemplateDetails';

// The index directs to the "About page"
class IndexRoute extends React.Component {
  render() {
    const { title, subtitle } = this.props.data.site.siteMetadata;
    const page = this.props.data.markdownRemark
    const { title: pageTitle, description: pageDescription } = page.frontmatter;
    const description = pageDescription !== null ? pageDescription : subtitle;

    return (
      <div>
        <Helmet>
          <title>{`${pageTitle} - ${title}`}</title>
          <meta name="description" content={description} />
        </Helmet>
        <PageTemplateDetails {...this.props} />
      </div>
    );
  }
}

export default IndexRoute;

// Get the "about me" page
export const pageQuery = graphql`
  query AboutQuery {
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
    markdownRemark(fields: { slug: { eq: "/about" } }) {
      id
      html
      frontmatter {
        title
        date
        description
      }
    }
  }
`;
