import React from 'react';

class TwitterFeed extends React.Component {
  // Ensure that twitter timeline reloads every time, since it can sometimes be spotty
  componentDidMount() {
    if (typeof twttr.widgets !== 'undefined') {
      twttr.widgets.load()
    }
  }

  render() {
    return (
      <div className="twitterfeed">
        <a className="twitter-timeline" data-width="340" data-height="400" data-theme="light"
           data-link-color="#2B7BB9" href="https://twitter.com/mihail_eric?ref_src=twsrc%5Etfw">
          Tweets by mihail_eric
        </a>
      </div>
    );
  }
}

export default TwitterFeed;
