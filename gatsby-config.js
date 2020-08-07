module.exports = {
  siteMetadata: {
    title: `Yugen`,
    name: `Yugen`,
    siteUrl: `https://jayantb1019.github.io/`,
    description: `Jayanth's blog on analytics`,
    hero: {
      heading: `Analytics Blog`,
      maxWidth: 750,
    },
    social: [
      {
        name: `twitter`,
        url: `http://twitter.com/JayanthBoddu/`,
      },
      {
        name: `github`,
        url: `https://github.com/jayantb1019`,
      },
      {
        name: `instagram`,
        url: `https://www.instagram.com/jayantb1019/`,
      },
      {
        name: `linkedin`,
        url: `https://www.linkedin.com/in/jayanthboddu/`,
      },
    ],
  },
  assetPrefix: "/blog",
  plugins: [
    {
      resolve: "@narative/gatsby-theme-novela",
      options: {
        contentPosts: "content/posts",
        contentAuthors: "content/authors",
        basePath: "/blog",
        authorsPage: true,
        sources: {
          local: true,
          // contentful: true,
        },
      },
    },
    {
      resolve: `gatsby-plugin-manifest`,
      options: {
        name: `Yugen by Jayanth Boddu`,
        short_name: `Yugen`,
        start_url: `/`,
        background_color: `#fff`,
        theme_color: `#fff`,
        display: `standalone`,
        icon: `src/assets/favicon.png`,
      },
    },
    {
      resolve: `gatsby-plugin-netlify-cms`,
      options: {},
    },
    {
      resolve: `gatsby-transformer-remark`,
      options: {
        plugins: [`gatsby-remark-autolink-headers`],
      },
    },
  ],
};
