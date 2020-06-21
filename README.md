# Sentiment-Extraction-using-Bert
<b>Using Bert to detect the sentiment of a given text and extract the words which contain the detected sentiment.</b><br>

### Methods Used
* Deep Learning
* Neural Networks
* Bert Base
* NLP

### Technologies
* Python, Spyder
* Pandas
* Numpy
* Pytorch
* Hugging Face - Tokenizer, Transformer
* Flask
* HTML

## Description
The aim of the project was to:
* Detect the sentiment of a given text. (Positive/Negative)
* Use the detected Sentiment and extract the words in the given text which convey the sentiment.

### <b> Process </b>
* First we trained a model using Bert to predict the sentiment for a given text on the twitter sentiment extraction data (See data sources). <br>
<b> Pre-Processing </b>
* The major task for this problem was to pre-process the data so as to make it work with the Bert Model. In the dataloader we take out input text, and first tokenize it using Bert Tokenizer and pass it to out sentiment predictor.
* We use the sentiment predicted along with the tokenized tweet and cerate out final tokenized model input where the sentiment token corresponds to the Question and the Tweet text is the context.
    `[CLS]Sentiment[SEP]Tweet[SEP]`
    
* We then define the span of our target or in other words the starting index and the ending index of the extracted words from the context (here tweet) which contain the sentiment detected.
* This whole process happens inside out dataloader and then passed on to our Bert Base model.

  <b> Post Processing </b>
* We get 2 output vectors from the model, one corresponds to the starting index and one to the ending index. (Note: the length of the vectors is equal to the max length provided by us).
* We then take the softmax of the outputs followed by argmax which gives us the final starting and ending index.
* Using the respective indices, the original tweet and the offsets provided by the Word Piece Tokenizer we convert these indices into out final extracted phrase. (Check out utils.py) <br>
<br>
We can see the end result as follows: <br>
! [Result](result.gif)
