from alpha_automl  import AutoMLClassifier
import pandas as pd
#from my_module import MyEmbedder, MySentimentEmbedder, MyBERTEmbedder, MySentenceEmbedder, MyTweetEmbedder

from transformers import AutoTokenizer, AutoModel
from alpha_automl.wrapper_primitives.huggingface import HuggingfaceInterface

if __name__ == '__main__':
    output_path = 'tmp/'
    train_dataset = pd.read_csv('/scratch/lm4428/d3m_latest/alpha-automl/examples/datasets/sentiment/train_data.csv')
    test_dataset = pd.read_csv('/scratch/lm4428/d3m_latest/alpha-automl/examples/datasets/sentiment/test_data.csv')

    target_column = 'sentiment'
    X_train = train_dataset.drop(columns=[target_column])
    y_train = train_dataset[[target_column]]

    automl = AutoMLClassifier(output_path, time_bound=40, verbose=True)



    #wrapper primitive
    #tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    #model = AutoModel.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment", output_hidden_states=True)
    model_name = 'cardiffnlp/twitter-roberta-base-sentiment'
    my_tweet_embedder = HuggingfaceInterface('cardiffnlp/twitter-roberta-base-sentiment', last_four_model_layers=False)
    
    my_sentiment_embedder = HuggingfaceInterface('allenai/reviews_roberta_base', last_four_model_layers=False)


    automl.add_primitives([(my_tweet_embedder, 'TEXT_ENCODER')])
    automl.add_primitives([(my_sentiment_embedder, 'TEXT_ENCODER')])
    automl.fit(X_train, y_train)
    automl.plot_leaderboard(use_print=True)
    
    
    #Adding from my module
    #my_embedder = MyEmbedder()
    #my_sentiment_embedder = MySentimentEmbedder()
    #my_bert_embedder = MyBERTEmbedder()
    #my_tweet_embedder = MyTweetEmbedder()
    #my_sentence_embedder = MySentenceEmbedder()
    #automl.add_primitives([(my_embedder, 'TEXT_ENCODER')])
    #automl.add_primitives([(my_sentiment_embedder, 'TEXT_ENCODER')])
    #automl.add_primitives([(my_bert_embedder, 'TEXT_ENCODER')])
    #automl.add_primitives([(my_sentence_embedder, 'TEXT_ENCODER')])
    #automl.add_primitives([(my_tweet_embedder, 'TEXT_ENCODER')])

    #automl.fit(X_train, y_train)

    #automl.plot_leaderboard(use_print=True)
    print("\n")

    X_test = test_dataset.drop(columns=[target_column])
    y_test = test_dataset[[target_column]]
    y_pred = automl.predict(X_test)
    print("y_pred: ")
    print(y_pred)
    print("\n")
    #print("automl.score(X_test, y_test) : {}".format(automl.score(X_test, y_test)))
    #automl.plot_comparison_pipelines()
