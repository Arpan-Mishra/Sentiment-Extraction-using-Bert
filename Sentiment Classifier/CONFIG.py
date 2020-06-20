<<<<<<< HEAD
import transformers

MAX_LEN = 100 
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 10 
EPOCHS = 5 
BERT_PATH = 'bert-base-uncased'
MODEL_PATH = 'model.bin'
TRAINING_FILE = 'C:/Users/Arpan/Downloads/Education/Project/Twitter Sentiment Classification/Data/train.csv'
TESTING_FILE = 'C:/Users/Arpan/Downloads/Education/Project/Twitter Sentiment Classification/Data/test.csv'
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH,do_lower_case = True)
=======
import transformers

MAX_LEN = 100 
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 10 
EPOCHS = 5 
BERT_PATH = 'bert-base-uncased'
MODEL_PATH = 'polarity_model.bin'
TRAINING_FILE = 'C:/Users/Arpan/Downloads/Education/Project/Twitter Sentiment Classification/Data/train.csv'
TESTING_FILE = 'C:/Users/Arpan/Downloads/Education/Project/Twitter Sentiment Classification/Data/test.csv'
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH,do_lower_case = True)
>>>>>>> 234f14c... Added app.py + final model
