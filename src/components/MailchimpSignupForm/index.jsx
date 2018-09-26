import React from 'react';
import addToMailchimp from 'gatsby-plugin-mailchimp';
import './style.scss';

const SIGN_UP_SUCCESS_ALERT_MESSAGE = 'Thanks for subscribing! Be on the lookout for AI blog posts, projects, code, and any other cool things I find. 🙂';
const SUCCESS = 'success';
const ALREADY_SUBSCRIBED_MAILCHIMP_FRAGMENT = 'is already subscribed to list';
const ALREADY_SUBSCRIBED_USER_MESSAGE = 'You have already been subscribed to my newsletter, but thanks for your enthusiasm!';

class MailchimpSignupForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      email: ''
    };
  }

  handleSubmit = (e) => {
    e.preventDefault();
    console.log('email: ', this.state.email);
    // listFields are optional if you are only capturing the email address.
    addToMailchimp(this.state.email, {})
      .then((data) => {
        // I recommend setting data to React state
        // but you can do whatever you want (including ignoring this `then()` altogether)
        console.log('Data: ', data);
        this.setState({ email: '' });
        // TODO (mihail): Add a nicer modal
        if (data.result === SUCCESS) {
          alert(SIGN_UP_SUCCESS_ALERT_MESSAGE);
        } else {
          // Display Mailchimp error message
          const errorMsg = data.msg;
          if (errorMsg.includes(ALREADY_SUBSCRIBED_MAILCHIMP_FRAGMENT)) {
            alert(ALREADY_SUBSCRIBED_USER_MESSAGE);
          } else {
            alert(errorMsg);
          }
        }
      })
      .catch(() => {
        // unnecessary because Mailchimp only ever
        // returns a 200 status code
        // see below for how to handle errors
      });
  }

  // As email input text area changes, update state with email
  handleChange = (e) => {
    this.setState({ email: e.target.value });
  }

  // Renders Twitter feed extracted from following: https://publish.twitter.com/?link_color=%232B7BB9&maxheight=200&maxwidth=280&query=%40mihail_eric&theme=light&widget=Timeline
  render() {
    return (
      <div className="mailchimp-signup">
        <div className="mailchimp-signup-prompt">
          I also run a newsletter with cool AI articles, projects, and code. Sign up to get updates!
        </div>
        <div className="mailchimp-signup-form">
          <form onSubmit={this.handleSubmit}>
            <label>
              Email
              <input className="mailchimp-signup-email-input" id="mailchimp-signup-email" type="text" name="mailchimp-signup-email" value={this.state.email} onChange={this.handleChange} />
            </label>
            <input className="mailchimp-signup-subscribe" type="submit" value="Subscribe" />
          </form>
        </div>
      </div>
    );
  }
}

export default MailchimpSignupForm;
