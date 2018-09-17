import React from 'react';
import './style.scss';

class PostSocialPanel extends React.Component {
  // Ensure that twitter timeline reloads every time, since it can sometimes be spotty
  // Also this https://www.stevenmercatante.com/how-to-add-twitter-buttons-to-a-gatsby-site/
  componentDidMount() {
    if (typeof twttr.widgets !== 'undefined') {
      twttr.widgets.load()
    }
  }

  // Renders Twitter feed extracted from following: https://publish.twitter.com/?link_color=%232B7BB9&maxheight=200&maxwidth=280&query=%40mihail_eric&theme=light&widget=Timeline
  render() {
    return (
      <div className="post-social-panel">
        <div className="post-social-panel-prompt">
          Like what you read? I would love to hear from you!
          <span className="post-social-panel-prompt-smiley"> 🙂</span>
        </div>
        <div className="post-social-panel-twitter">
          {/* Tweet button */}
          <span>
            <a className="twitter-share-button"
               href="https://twitter.com/intent/tweet?text=@mihail_eric"
               data-size="large">
              Tweet
            </a>
          </span>
          <span className="post-social-panel-divider" />
          {/* Follow button */}
          <span>
            <a className="twitter-follow-button"
               href="https://twitter.com/mihail_eric"
               data-show-count="false"
               data-show-screen-name="false"
               data-size="large">
              Follow
            </a>
          </span>
        </div>
      </div>
    );
  }
}

export default PostSocialPanel;
