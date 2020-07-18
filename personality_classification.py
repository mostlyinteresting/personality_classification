
import re
import numpy as np
import pandas as pd
import sqlite3
#import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
#from sklearn.manifold import TSNE
from scipy import stats

import spacy
import en_core_web_sm
import contractions
sp = en_core_web_sm.load()

stop_words = sp.Defaults.stop_words

random_state = 12345

hashtags = re.compile(r"^#\S+|\s#\S+")
mentions = re.compile(r"^@\S+|\s@\S+")
urls = re.compile(r"https?://\S+")    

#built from 'setup_reddit_data.py' script
conn = sqlite3.connect('reddit.sqlite')

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.2f}; misclass={:0.2f}'.format(accuracy, misclass))
    plt.show()
    
def process_text(text):
    text = hashtags.sub(' hashtag', text)
    text = mentions.sub(' entity', text)
    text = urls.sub(' website', text)
    text = re.sub(r"[^A-Za-z0-9(),!.?\'\`]", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r":", " ", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ) ", text)
    text = re.sub(r"\?", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.split()
    text = [ contractions.expandContractions(x) for x in text]
    text = sp(' '.join([ word for word in text if not word in stop_words]))
    text = ' '.join([ word.lemma_ for word in text ])
    
    return text.strip().lower()


if __name__ == '__main__':
    
    yVar = '_label'
    test_size = 0.2
    
    data = pd.read_sql_query("SELECT * FROM data", conn)
    data['subreddit_name_prefixed'] = data['subreddit_name_prefixed'].str.replace('r/','')
    X = data[ data['subreddit_name_prefixed'].isin(['extroverts','introvert']) ]
    X = X[ (X['distinguished'] != 'moderator') & (X['selftext'] != '') ]
    X['selftext'] = X['selftext'].apply(process_text)

    X['value'] = pd.Categorical(X['subreddit_name_prefixed'])
    yKey = dict(zip(X['value'].cat.categories, X['value'].cat.codes))
    X[yVar] = X['value'].cat.codes
    X.drop('value', axis = 1, inplace = True)
    
    vectorizer = TfidfVectorizer(ngram_range = (1,3), max_features = 3000)
    X0 = vectorizer.fit_transform(X['selftext'])
    features = vectorizer.get_feature_names()
    X0 = pd.DataFrame(X0.toarray(), columns = features)
    
    # X_embedded = TSNE(n_components=2).fit_transform(X0)
    # X_embedded = pd.DataFrame(X_embedded, columns = ['dim1','dim2'])
    # X_embedded[yVar] = X[yVar].values
    # sns.scatterplot(x="dim1", y="dim2", hue=yVar, data=X_embedded)
    
    X0[yVar] = X[yVar].values
    
    train, test = train_test_split(X0, test_size=test_size, random_state=random_state)
    
    lr = LogisticRegression(random_state = random_state)
    lr.fit(train[features], train[yVar])
    test_pred_lr = lr.predict(test[features])
    
    print(accuracy_score(test[yVar], test_pred_lr))
    print(classification_report(test[yVar], test_pred_lr))
    
    svm = LinearSVC(random_state = random_state)
    svm.fit(train[features], train[yVar])
    test_pred_svm = svm.predict(test[features])
    
    print(accuracy_score(test[yVar], test_pred_svm))
    print(classification_report(test[yVar], test_pred_svm))
    
    '''
    Myers-Briggs Types
    '''
    
    mbTypes = ['INTJ','INTP','ENTJ','ENTP',
               'INFJ','INFP','ENFJ','ENFP',
               'ISTJ','ISFJ','ESTJ','ESFJ',
               'ISTP','ISFP','ESTP','ESFP']
    mbTypes = [ x.lower() for x in mbTypes ]
    
    # mbIntrovertTypes = [ x for x in mbTypes if x.startswith('i') ]
    # mbExtrovertTypes = [ x for x in mbTypes if x.startswith('e') ]
    
    data['subreddit_name_prefixed'] = data['subreddit_name_prefixed'].str.lower()
    data['subreddit_name_prefixed'] = data['subreddit_name_prefixed'].str.replace('estj2','estj')
    
    missingTypes = [ x for x in mbTypes if x not in data['subreddit_name_prefixed'].unique().tolist() ]
    
    X = data[ data['subreddit_name_prefixed'].isin(mbTypes) ]
    X = X[ (X['distinguished'] != 'moderator') & (X['selftext'] != '') ]
    X['selftext'] = X['selftext'].apply(process_text)
    X['value'] = pd.Categorical(X['subreddit_name_prefixed'])
    yKey = dict(zip(X['value'].cat.categories, X['value'].cat.codes))
    X[yVar] = X['value'].cat.codes
    X.drop('value', axis = 1, inplace = True)
    
    vectorizer = TfidfVectorizer(ngram_range = (1,3), max_features = 10000)
    X0 = vectorizer.fit_transform(X['selftext'])
    features = vectorizer.get_feature_names()
    X0 = pd.DataFrame(X0.toarray(), columns = features)
    
#    X_embedded = TSNE(n_components=2).fit_transform(X0.drop(yVar, axis = 1))
#    X_embedded = pd.DataFrame(X_embedded, columns = ['dim1','dim2'])
#    X_embedded[yVar] = X[yVar].values
#    sns.scatterplot(x="dim1", y="dim2", hue=yVar, data=X_embedded)

    X0[yVar] = X[yVar].values
    
    train, test = train_test_split(X0, test_size=test_size, random_state=random_state)
    
    lr = LogisticRegression()
    lr.fit(train[features], train[yVar])
    test_pred_mb_lr = lr.predict(test[features])
    
    print(classification_report(test[yVar], test_pred_mb_lr))
    
    svm = LinearSVC()
    svm.fit(train[features], train[yVar])
    test_pred_mb_svc = svm.predict(test[features])
    
    print(classification_report(test[yVar], test_pred_mb_svc))

    f1 = make_scorer(f1_score,average = 'macro')
    
    svmCV = LinearSVC()
    params = {"C": stats.uniform(2, 10)}
    randSearch = RandomizedSearchCV(svmCV, param_distributions = params, n_iter = 20, n_jobs = 4,
                                     cv = 3, random_state = random_state, scoring = f1)
    randSearch.fit(train[features], train[yVar])
    svmBest = randSearch.best_estimator_
    svmBest.fit(train[features], train[yVar])
    test_pred_mb_svc_best = svmBest.predict(test[features])
    
    print(classification_report(test[yVar], test_pred_mb_svc_best))

    cm = confusion_matrix(test[yVar], test_pred_mb_svc_best)
    target_names = test.groupby(yVar)['subreddit_name_prefixed'].first().to_dict().values()
    target_names = list(target_names)
    plot_confusion_matrix(cm, target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True)