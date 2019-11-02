runfile('./data/process_data.py', args='./data/disaster_messages.csv ./data/disaster_categories.csv ./data/DisasterResponse.db')
runfile('./models/train_classifier.py',args='./data/DisasterResponse.db ./models/resulting_model.pkl')
http://127.0.0.1:5000/index

best fit:
{'clf__estimator__min_samples_split': 2, 'clf__estimator__n_estimators': 200, 'features__text_pipeline__vect__max_df': 0.5, 'features__text_pipeline__vect__ngram_range': (1, 2), 'features__transformer_weights': {'text_pipeline': 1, 'starting_verb': 0.5}}