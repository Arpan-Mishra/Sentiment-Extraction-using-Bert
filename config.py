<<<<<<< HEAD
import tokenizers

MAX_LEN = 100
TRAIN_BATCH_SIZE = 16 
VALID_BATCH_SIZE = 10 
EPOCHS = 5
BERT_PATH = 'bert-base-uncased'
MODEL_PATH = 'model2.bin'
TRAINING_FILE = 'C:/Users/Arpan/Downloads/Education/Project/Twitter Sentiment Classification/Data/train.csv'
TOKENIZER = tokenizers.BertWordPieceTokenizer(
        BERT_PATH+'-vocab.txt'        
        )
=======
import tokenizers
import transformers

MAX_LEN = 100
TRAIN_BATCH_SIZE = 16 
VALID_BATCH_SIZE = 10 
TEST_BATCH_SIZE = 5
EPOCHS = 5
BERT_PATH = 'bert-base-uncased'
MODEL_PATH = 'extraction_model.bin'
TRAINING_FILE = 'C:/Users/Arpan/Downloads/Education/Project/Twitter Sentiment Classification/Data/train.csv'
TESTING_FILE = 'C:/Users/Arpan/Downloads/Education/Project/Twitter Sentiment Classification/Data/test.csv'
SENTIMENT_MODEL_PATH = 'C:/Users/Arpan/Downloads/Education/Project/Twitter Sentiment Classification/Text - Extraction/Sentiment Classifier/polarity_model.bin'
TOKENIZER = tokenizers.BertWordPieceTokenizer(
        BERT_PATH+'-vocab.txt', lowercase=True        
        )
Plrty_Tokenizer = transformers.BertTokenizer.from_pretrained(BERT_PATH,do_lower_case=True)
>>>>>>> 234f14c... Added app.py + final model
