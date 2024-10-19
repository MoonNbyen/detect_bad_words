
# Project: *Am I the A-hole? A Moral Classification of Reddit Posts*

This project aims to classify posts from the Reddit community "Am I the A-hole?" (AITA) using machine learning models. The objective is to determine whether the poster is perceived as an "Asshole" or "Not the A-hole" based on the content of their post. The process involves data extraction, cleaning, topic modeling, and classification using various machine learning algorithms. 
### Link to the code: https://github.com/MoonNbyen/detect_bad_words/blob/main/Dataminingfinal.ipynb

#### Key Components:

1. **Data Collection**:  
   - Utilized Reddit’s API to scrape posts from the AITA subreddit. The data collected includes titles, self-texts (body of the post), and verdicts (e.g., "Asshole", "Not the A-hole", "Everyone Sucks", etc.).

2. **Data Preprocessing**:  
   - Cleaned and prepared the data for analysis by handling missing values, removing unnecessary categories, and processing textual data through tokenization and stopword removal. 
   - Grouped similar verdicts for balanced classification into two categories: "Asshole" and "Not the A-hole."

3. **Text Analysis**:  
   - **Topic Modeling**: Applied Latent Dirichlet Allocation (LDA) to post titles to extract common topics discussed within the subreddit. 
   - **Text Classification**: Trained multiple machine learning models (Naive Bayes, Support Vector Classifier, Random Forest, Logistic Regression, and Neural Networks) on the pre-processed text data to predict the moral classification of new posts.

4. **Model Evaluation**:  
   - Evaluated models using accuracy scores, confusion matrices, and cross-validation to select the best-performing model. 
   - Conducted hyperparameter tuning with GridSearchCV for optimal performance.

5. **Interactive Demo**:  
   - Developed an interactive demo using Gradio, allowing users to input text and receive predictions on whether the poster is the "Asshole" or "Not the A-hole."

#### Technologies Used:
- **APIs**: Reddit API for data extraction.
- **Data Processing**: Pandas, Numpy, Gensim (for topic modeling), NLTK (for text preprocessing).
- **Machine Learning**: Scikit-learn (for classification models and pipelines), Random Forest, Logistic Regression, Support Vector Machines.
- **Visualization**: Seaborn and Matplotlib for displaying data distributions and word clouds.
- **Deployment**: Gradio for building an interactive web interface.

This project demonstrates the application of sentiment analysis, moral classification, and text-based machine learning modeling on real-world social media data.



```markdown
# Am I the A-hole? A Moral Classification 

This project aims to classify Reddit posts from the subreddit r/AITA (Am I the A-hole) based on user verdicts. The classification task involves building machine learning models to predict whether a poster is considered an "Asshole" or "Not the A-hole."

## Getting Data from Reddit API

The data is scraped from the Reddit API using the following code:

<details><summary><u>Reddit Scraper</u></summary>
<p>

```python
import requests

app_id = 'your_app_id'
secret = 'your_secret'
auth = requests.auth.HTTPBasicAuth(app_id, secret)

headers = {'User-Agent': 'AITA_Classifier/0.1'}
res = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=headers)

token = res.json()['access_token']
headers['Authorization'] = 'bearer {}'.format(token)

res = requests.get('https://oauth.reddit.com/r/AITAFiltered/new', headers=headers, params={'limit': '50000'})
result = res.json()
```

The scraped data is saved as a JSON file for further analysis.

</p>
</details>

## Data Preprocessing

1. The dataset contains posts classified into categories like "Asshole," "Not the A-hole," "Everyone Sucks," and "No A-holes Here."
2. For simplicity, we merge "Everyone Sucks" with "Asshole" and "No A-holes Here" with "Not the A-hole."
   
```python
df.loc[df['verdict'] == 'Everyone Sucks', 'verdict'] = 'Asshole'
df.loc[df['verdict'] == 'No A-holes here', 'verdict'] = 'Not the A-hole'
```

## Data Cleaning and Text Preprocessing

We perform basic text preprocessing, including removing contractions and stopwords, and tokenizing the title for topic modeling.

```python
df['title'] = df['title'].apply(lambda x: x.replace("'m"," am").replace("n't"," not").replace("'ve"," have").replace("'s"," is"))
df['title_tokenized'] = df['title'].apply(lambda x: tokenizer.tokenize(x.lower()))
```

## Topic Modeling

The LDA model is used to identify latent topics in the titles of the posts.

```python
from gensim import corpora, models

aitadict = corpora.Dictionary(df.title_tokenized.tolist())
aitacorp = [aitadict.doc2bow(x) for x in df.title_tokenized.tolist()]

aitamodel = models.LdaModel(corpus=aitacorp, id2word=aitadict, num_topics=10, passes=10)
```

## Classification

We use different models to classify the posts into "Asshole" or "Not the A-hole," including:

- Naive Bayes
- Support Vector Classification
- Random Forest
- Logistic Regression
- Perceptron

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

pipelineMNB = Pipeline([
    ('vect', CountVectorizer(preprocessor=text_preprocessor)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

pipelineMNB.fit(ASS_train, label_train)
predictions = pipelineMNB.predict(ASS_test)
print(classification_report(label_test, predictions))
```

## Demo

A Gradio app has been created to test the classification models. Enter a text to see if the model predicts the user as "Asshole" or "Not the A-hole."

```python
import gradio as gr

def show(text):
    a = [text]
    if pipeline_svc.predict(a)[0] == 1:
        return "You are the asshole"
    else:
        return "You're not the asshole"

iface = gr.Interface(fn=show, inputs="text", outputs="text")
iface.launch(share=True)
```
