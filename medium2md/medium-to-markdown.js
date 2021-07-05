const mediumToMarkdown = require('medium-to-markdown');

medium_post_url = 'https://medium.com/@jayanth.boddu.91/machine-learning-ai-for-predictive-exploration-e11c41439aa'
// Enter url here
mediumToMarkdown.convertFromUrl(medium_post_url)
.then(function (markdown) {
  console.log(markdown); //=> Markdown content of medium post
});

// RUN THIS FILE WITH MEDIUM POST URL USING 
// node medium-to-markdown.js >> file.md 