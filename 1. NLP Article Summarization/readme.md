## NLP Article Summarization Using Transformers

In this notebook, I used the Beautiful Soup package to extract the text from a web article. Then I used the NLTK and textblob libraries to preprocess the text by removing words that don't contribute much to the overall interpretation of the text, as well as applying a lemma-ization process to simplify words to their rawest meaning. Then, I chunked up the text into batch sizes that were then fed through the Hugging Face Text-Summarization library. Lastly, I re-ran the initial summary through the pipeline a second time which allowed for an even more condensed summary.

This article was not chosen for any special content, but for the simplicity of the html formatting, which allowed for the easy text extraction. This process could be automated and scheduled to summarize articles and store them for later reading.

One thing I am considering for future work on this is to add Text to Speech pipeline to read the summaries out loud. Another future project will be scheduling a scraping event (something like headlines only), and running sentiment analysis on those.
