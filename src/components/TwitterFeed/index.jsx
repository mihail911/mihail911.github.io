import React from 'react';

class TwitterFeed extends React.Component {
  // Ensure that twitter timeline reloads every time, since it can sometimes be spotty
  componentDidMount() {
    if (typeof twttr.widgets !== 'undefined') {
      twttr.widgets.load()
    }
  }

  // Renders Twitter feed extracted from following: https://publish.twitter.com/?link_color=%232B7BB9&maxheight=200&maxwidth=280&query=%40mihail_eric&theme=light&widget=Timeline
  render() {
    return (
      <div className="twitterfeed">
        <a className="twitter-timeline" data-width="400" data-height="400" data-theme="light"
           data-link-color="#2B7BB9" href="https://twitter.com/mihail_eric?ref_src=twsrc%5Etfw">
          Tweets by mihail_eric
        </a>
      </div>
    );
  }
}

export default TwitterFeed;
