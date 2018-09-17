import React from 'react';
import Link from 'gatsby-link';
import moment from 'moment';
import Disqus from '../Disqus/Disqus';
import PostSocialPanel from '../PostSocialPanel';
import './style.scss';

class PostTemplateDetails extends React.Component {
  render() {
    const { subtitle, author } = this.props.data.site.siteMetadata;
    const post = this.props.data.markdownRemark;
    const tags = post.fields.tagSlugs;
    const relatedPosts = post.frontmatter.related;

    const homeBlock = (
      <div className="post-single__sidebar" >
        <div className="post-single__sidebar-all-articles">
          <Link to="/" className="post-single__sidebar-all-articles-link">Back to Home</Link>
        </div>
        <div className="post-single__sidebar__related-posts-title">
          Related Articles
        </div>
        <div className="post-single__sidebar__related-posts-items">
          <ul>
          {
            relatedPosts.map((relatedPost) => {
              const [title, postSlug] = relatedPost.split(':');
              return (
                <li key={title}>
                  <Link key={title} to={postSlug} className="post-single__sidebar__related-posts-items-link">
                    {title}
                  </Link>
                </li>
              );
            })
          }
          </ul>
        </div>
      </div>
    );

    const tagsBlock = (
      <div className="post-single__tags">
        <ul className="post-single__tags-list">
          {tags && tags.map((tag, i) => (
            <li className="post-single__tags-list-item" key={tag}>
              <Link to={tag} className="post-single__tags-list-item-link">
                {post.frontmatter.tags[i]}
              </Link>
            </li>
          ))}
        </ul>
      </div>
    );

    const commentsBlock = (
      <div>
        <Disqus postNode={post} siteMetadata={this.props.data.site.siteMetadata} />
      </div>
    );

    return (
      <div className="post-full-wrapper">
        {homeBlock}
        <div className="post-single">
          <div className="post-single__inner">
            <h1 className="post-single__title">{post.frontmatter.title}</h1>
            <div className="post-single__date">
              <em>{moment(post.frontmatter.date).format('MMMM YYYY')}</em>
            </div>
            <div className="post-single__body" dangerouslySetInnerHTML={{ __html: post.html }} />
          </div>
          <div className="post-single__footer">
            {tagsBlock}
            <hr />
            {/* TODO (mihail): Do something better with the footer. Link to newsletter?
            Link to social media? */}
            <p className="post-single__footer-text" />
            <PostSocialPanel />
            {commentsBlock}
          </div>
        </div>
      </div>
    );
  }
}

export default PostTemplateDetails;
