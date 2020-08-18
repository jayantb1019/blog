module.exports = {
  pathPrefix: "blog",
  siteMetadata: {
    title: `Yugen`,
    name: `Yugen`,
    siteUrl: `https://analytics-blog-d3223.web.app`,
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
  assetPrefix: "/",
  plugins: [
    {
      resolve: "@narative/gatsby-theme-novela",
      options: {
        contentPosts: "content/posts",
        contentAuthors: "content/authors",
        basePath: "/",
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
        plugins: [
          `gatsby-remark-autolink-headers`,
          {
            resolve: "gatsby-remark-external-links",
            options: {
              target: "_blank",
              rel: "nofollow",
            },
          },
        ],
      },
    },
    {
      resolve: "gatsby-plugin-firebase",
      options: {
        credentials: {
          apiKey: "AIzaSyCyPIhmGxHzbe-xTDei6JfXhpWrfsr92X4",
          authDomain: "analytics-blog-d3223.firebaseapp.com",
          databaseURL: "https://analytics-blog-d3223.firebaseio.com",
          projectId: "analytics-blog-d3223",
          storageBucket: "analytics-blog-d3223.appspot.com",
          messagingSenderId: "828646494157",
          appId: "1:828646494157:web:518a9347f032e02979be84",
          measurementId: "G-C0BEP3XC72",
        },
      },
    },
  ],
};
