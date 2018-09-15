import React from 'react';
import Link from 'gatsby-link';
import moment from 'moment';
import TagButton from '../TagButton';
import './style.scss';

import { findMatchingTagSlug } from '../../util/tagUtils';

class Post extends React.Component {
  render() {
    const { title, date, category, description, tags } = this.props.data.node.frontmatter;
    const { slug, categorySlug, tagSlugs } = this.props.data.node.fields;
    return (
      <div className="post">
        <div className="post__meta">
          <time className="post__meta-time" dateTime={moment(date).format('MMMM D, YYYY')}>
            {moment(date).format('MMMM YYYY')}
          </time>
          <span className="post__meta-divider" />
          <span className="post__meta-category" key={categorySlug}>
            {
              // Add tag button for every tag for given post
              tags.map((tag) => {
                let matchingTagSlug = findMatchingTagSlug(tag, tagSlugs);
                // Couldn't find matching tag slug so default to first tag slug, rather than
                // throwing error
                if (matchingTagSlug == null) {
                  matchingTagSlug = tagSlugs[0];
                  console.log('Matching tag not found');
                }
                return (
                  <span key={tag}>
                    <TagButton
                      key={tag}
                      tagSlug={matchingTagSlug}
                      tag={tag}
                    />
                    <span className="post__meta-mini-spacer" />
                  </span>
                );
              })

            }
          </span>
        </div>
        <h2 className="post__title">
          <Link className="post__title-link" to={slug}>{title}</Link>
        </h2>
        <p className="post__description">{description}</p>
      </div>
    );
  }
}

export default Post;
