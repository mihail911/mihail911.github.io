module.exports = {
  siteMetadata: {
    url: 'https://mihaileric.com',
    title: 'Mihail Eric',
    subtitle: 'Engineer, researcher, educator.',
    disqusShortname: 'mihaileric',
    menu: [
      {
        label: 'Blog',
        path: '/blog/'
      },
      {
        label: 'Projects',
        path: '/projects/'
      },
      {
        label: 'Teaching',
        path: '/teaching/'
      }
    ],
    author: {
      name: 'Mihail Eric',
      email: 'mihail911@gmail.com',
      twitter: 'mihail_eric',
      github: 'mihail911',
      linkedin: 'mihaileric'
    },
    copyright: 'Copyright'
  },
  plugins: [
    {
      resolve: 'gatsby-source-filesystem',
      options: {
        path: `${__dirname}/src/pages`,
        name: 'pages'
      }
    },
    {
      resolve: 'gatsby-transformer-remark',
      options: {
        plugins: [
          {
            resolve: 'gatsby-remark-images',
            options: {
              maxWidth: 960,
              showCaptions: true
            }
          },
          {
            resolve: 'gatsby-remark-responsive-iframe',
            options: { wrapperStyle: 'margin-bottom: 1.0725rem' }
          },
          'gatsby-remark-prismjs',
          'gatsby-remark-copy-linked-files',
          'gatsby-remark-smartypants',
          'gatsby-remark-katex'
        ]
      }
    },
    'gatsby-transformer-sharp',
    'gatsby-plugin-sharp',
    {
      resolve: 'gatsby-plugin-google-analytics',
      options: { trackingId: 'UA-120722682-1' }
    },
    {
      resolve: `gatsby-plugin-google-fonts`,
      options: {
        fonts: [`Alegreya\:400,400i,700,700i`, `Alegreya Sans\:500`]
      }
    },
    {
      resolve: 'gatsby-plugin-sitemap',
      options: {
        query: `
            {
              site {
                siteMetadata {
                  url
                }
              }
              allSitePage(
                filter: {
                  path: { regex: "/^(?!/404/|/404.html|/dev-404-page/)/" }
                }
              ) {
                edges {
                  node {
                    path
                  }
                }
              }
          }`,
        output: '/sitemap.xml',
        serialize: ({ site, allSitePage }) =>
          allSitePage.edges.map((edge) => {
            return {
              url: site.siteMetadata.url + edge.node.path,
              changefreq: 'daily',
              priority: 0.7
            };
          })
      }
    },
    {
      resolve: 'gatsby-plugin-mailchimp',
      options: {
        endpoint: 'https://mihaileric.us19.list-manage.com/subscribe/post?u=62eb9837d59164bc45efac84e&amp;id=b3f700cf59'
      }
    },
    'gatsby-plugin-offline',
    'gatsby-plugin-catch-links',
    'gatsby-plugin-react-helmet',
    'gatsby-plugin-postcss-sass'
  ]
};
