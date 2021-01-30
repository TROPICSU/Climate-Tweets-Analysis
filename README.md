# Climate-Tweets-Analysis
ASHNA'S CODES -
1. BAG OF WORDS MODEL USING LOGISTIC REGRESSION ALGO FOR CLASSIFICATION - 
In this model, I have preprocessed tweets using regex and then lemmatized them.In basic analysis of raw data, we create a pie chart to see the distribution of tweets in Yes, No and N/A format.Later on we will use this pie chart for comparative analysis after training nlp model. Since the dataset was lacking labels I added the same using Textblob library using sentiment function from the library.This helped us classify the tweets in 3 classes - negative , positive and neutral.After that we used CountVectorizer to change the tweet tokens into vectors which ultimately ends up as a sparse matrix of 4134 rows and 2767 features.Thereafter we split the dataset into training andd test sets for training the model and for validating it.
In part 2 , we train the logistic regression model over the training set. 
Then finally in part 3 , we predict values using x_test array. Now we will evaluate the model using f1_score metric which gives us the best value for true negative, true positives from the dataset. In this we got f1_score to be 65%. 
After this , we plot another pie chart to visualize the distinct differences in text classification of tweets. End results being that the Positive tweets reduced from  50.8% to 35% which is a drop of 15%. For the Negative tweets we see an increase of 5% from 18.5% to 23.2%. When it comes to Neutral tweets we can see an 11% increase from 30.7% to 41.8%. Neutral tweets also form the biggest category out of the three followed by positive and then negative tweets.
So, we can see that there is a lot of ambuiguity about Global Warming , followed by sheer confidence regarding Climate Change in people, with few people refuting it's existence altogether.
2.BERT ALGORITHM -
Here we use 'ktrain' which is a library to help build, train, debug, and deploy neural networks in the deep learning software framework. Keras.ktrain is designed to work seamlessly with Keras. Here, we load data and define a model just as you would normally do in Keras. The following code was copied directly from the Keras text classification example. It loads the  dataset and defines a simple text classification model to infer the sentiment of tweets regarding Global Warming.
To use ktrain, we simply wrap our model and data in a ktrain.Learner object using the get_learner function.
The default batch size is 32, but this can be changed by supplying a batch_size argument to get_learner. The Learner object facilitates training your neural network in various ways. For instance, invoking the fit method of the Learner object allows you to train interactively at different learning rates.
The fit_onecycle method increases the learning rate from a base rate to a maximum rate for the first half of training and decays the learning rate to a near-zero value for the second half of training. 
We run 4 epochs and we can see that the accuracy for the model improves with almost 30%  per epoch. It goes from 44% to 72% and then to a staggering 92% in the 3rd epoch itself. The 4th epoch however gives us an amazing accuracy of 98% !
Thus BERT's claim at being the best NLP algorithm holds true!

**Naveen's Code** :

Text classification and anlysis using TF-IDF measurement and Prediction through NB,SVM,Logistic Regression Models

Code contains:

1.Making changes in the 'existence' column of data converting similar meaning words to a single word(Ex: Yes,Y,YES to yes)

2.Visualizing the data categorically using a pie chart

3.Data preprocessing : removing URL'S,remove HTML tags,removing Punctuations,Tokenizing the words,removing stopwords,lemmatizing,stemming.

4.Analysis after Text preprocessing : Visualization of most used words in Positive and Negative tweets through wordcoud and   Bargraph

5.Using TF-IDF measurement for vectorizing the every words and for getting frequency scores for words

6.Prediction using Classification methods :NaiveBayes, SVM and Logistic Regression.

7.Analysing the predictions using Precision ,Recall f1-score and Confusion Matrix .




