import React from 'react';
import { Set } from 'immutable';
import Link from 'gatsby-link';
import moment from 'moment';
import PostSocialPanel from '../PostSocialPanel';
import MailchimpSignupForm from '../MailchimpSignupForm'
import './style.scss';

const MAX_RELATED_POSTS_TO_SHOW = 5;

class PostTemplateDetails extends React.Component {
  render() {
    const post = this.props.data.markdownRemark;
    // Get collection of all tags for given post
    const postTags = Set(post.frontmatter.tags);
    const tags = post.fields.tagSlugs;

    const allPosts = this.props.data.allMarkdownRemark.edges;
    // Get all related posts that have some tags in common with current post (except post itself)
    const relatedPosts = allPosts.map((currPost) => {
      const tagIntersect = postTags.intersect(currPost.node.frontmatter.tags);
      if (tagIntersect.size > 0 && (currPost.node.frontmatter.title !== post.frontmatter.title)) {
        return [currPost.node.frontmatter.title, currPost.node.fields.slug];
      }
      return null;
    }).filter((elem) => {
       return elem !== null;
    });
    const homeBlock = (
      <div className="post-single__sidebar" >
        <div className="post-single__sidebar-all-articles">
          <Link to="/" className="post-single__sidebar-all-articles-link">Back to Home</Link>
        </div>
        {
          relatedPosts.length > 0 ?
            <div className="post-single__sidebar__related-posts-title">
              Related Articles
            </div> : null
        }
        <div className="post-single__sidebar__related-posts-items">
          <ul>
          {
            // Display related posts in sidebar
            relatedPosts.slice(0, MAX_RELATED_POSTS_TO_SHOW).map((relatedPost) => {
              const [title, postSlug] = relatedPost;
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
            <MailchimpSignupForm />
          </div>
        </div>
      </div>
    );
  }
}

export default PostTemplateDetails;
