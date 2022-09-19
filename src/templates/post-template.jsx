import React from 'react';
import Helmet from 'react-helmet';
import PostTemplateDetails from '../components/PostTemplateDetails';

class PostTemplate extends React.Component {
  render() {
    const { title, subtitle } = this.props.data.site.siteMetadata;
    const post = this.props.data.markdownRemark;
    const { title: postTitle, description: postDescription } = post.frontmatter;
    const description = postDescription !== null ? postDescription : subtitle;
    const twitterCardUrl = post.frontmatter.img_url;
    const DEFAULT_IMG_URL = "https://cdn.searchenginejournal.com/wp-content/uploads/2021/08/20-awesome-sources-of-free-data-612c26257336f-sej.jpg";
    return (
      <div>
        <Helmet>
          <title>{`${postTitle} - ${title}`}</title>
          <meta name="description" content={description} />
          <meta name="twitter:card" content="summary_large_image" />
          <meta name="twitter:title" content={postTitle} />
          <meta name="twitter:image" content={twitterCardUrl ? twitterCardUrl: DEFAULT_IMG_URL} />
        </Helmet>
        <PostTemplateDetails {...this.props} />
      </div>
    );
  }
}

export default PostTemplate;

export const pageQuery = graphql`
  query PostBySlug($slug: String!) {
    site {
      siteMetadata {
        title
        subtitle
        copyright
        author {
          name
          twitter
        }
        url
      }
    }
    allMarkdownRemark(
        limit: 1000,
        filter: { frontmatter: { layout: { eq: "post" }, draft: { ne: true } } },
        sort: { order: DESC, fields: [frontmatter___date] }
      ){
      edges {
        node {
          fields {
            slug
            tagSlugs
          }
          frontmatter {
            title
            tags
            date
          }
        }
      }
    }
    markdownRemark(fields: { slug: { eq: $slug } }) {
      id
      html
      fields {
        tagSlugs
      }
      frontmatter {
        title
        tags
        date
        description
        img_url
      }
    }
  }
`;
