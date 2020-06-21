# Sentiment-Extraction-using-Bert
<b>Using Bert to detect the sentiment of a given text and extract the words which contain the detected sentiment.</b><br>

## Table of Contents
  * [Methods Used](#methods-used)
  * [Technologies](#technologies)  
  * [Description](#description)
  * [Model Building](#model-building)
  * [Credits](#credits)
  * [Contact](#contact)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>


## Methods Used
* Deep Learning
* Neural Networks
* Bert Base
* NLP

## Technologies
* Python, Spyder
* Pandas
* Numpy
* Pytorch
* Hugging Face - Tokenizer, Transformer
* Flask
* HTML

#### For web framework requirements : `pip install -r requirements.txt`

## Description
The aim of the project was to:
* Detect the sentiment of a given text. (Positive/Negative)
* Use the detected Sentiment and extract the words in the given text which convey the sentiment.

## Model Building
* First we trained a model using Bert to predict the sentiment for a given text on the twitter sentiment extraction data (See data sources). <br>

    <b> Pre-Processing </b>
* The major task for this problem was to pre-process the data so as to make it work with the Bert Model. In the dataloader we take our input text, and first tokenize it using Bert Tokenizer and pass it to out sentiment predictor.
* We use the sentiment predicted along with the tokenized tweet and create our final tokenized model inputs where the sentiment token corresponds to the Question and the Tweet text is the context.
    `[CLS]Sentiment[SEP]Tweet[SEP]`
    
* We then define the span of our target or in other words the starting index and the ending index of the extracted words from the context (here tweet) which contain the sentiment detected.
* This whole process happens inside our dataloader and then passed on to our Bert Base model.

  <b> Post Processing </b>
* We get 2 output vectors from the model, one corresponds to the starting index and one to the ending index. (Note: the length of the vectors is equal to the max length provided by us).
* We then take the softmax of the outputs followed by argmax which gives us the final starting and ending index.
* Using the respective indices, the original tweet and the offsets provided by the Word Piece Tokenizer we convert these indices into our final extracted phrase. (Check out utils.py) <br>
<br>

<b> Productionisation </b>
* In this step, I built a flask API endpoint that was hosted on a local webserver.

We can see the end result as follows: <br>
<img src= "https://github.com/Arpan-Mishra/Sentiment-Extraction-using-Bert/blob/master/result.gif">

## Credits
* Data Source: https://www.kaggle.com/c/tweet-sentiment-extraction
* Bert Concept: http://jalammar.github.io/illustrated-bert/
* Bert Text Extraction Tutorial: https://www.youtube.com/watch?v=XaQ0CBlQ4cY
* Flask tutorial: https://towardsdatascience.com/predicting-reddit-flairs-using-machine-learning-and-deploying-the-model-using-heroku-part-3-c3cd19374596

## Contact
* For any queries and feedback please contact me at mishraarpan6@gmail.com

Note: The project is only for education purposes, no plagiarism is intended.
