import json
import plotly
import pandas as pd
import re
import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
    
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from functions import tokenize

from app import app, df, model
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # create visuals
    
    # Graph 1
    graph_one = []   
    label_counts = df.groupby('label').count()['cleaned_tweet']
    label_names = list(label_counts.index)

    graph_one.append(
        Bar(
            x=label_names,
            y=label_counts
        )
    )

    layout_one = dict(title = 'Distribution of Message Labels',
        xaxis = dict(title = 'Label'),
        yaxis = dict(title = 'Count'),
    )
    
    # Graph 2

    graph_two = []  

    sw = stopwords.words("english")
    text = df.cleaned_tweet.str.cat(sep=' ')
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    str_list = text.split(" ")
    s = pd.Series(str_list)
    s = s[s != ""]
    s = s[~s.isin(sw)]

    word_counts = s.value_counts()[:10].tolist()
    word_names = s.value_counts()[:10].index.tolist()

    graph_two.append(
        Bar(
            x=word_names,
            y=word_counts
        )
    )

    layout_two = dict(title = 'Top 10 Common Words',
        xaxis = dict(title = 'Words'),
        yaxis = dict(title = 'Count'),
    )
    
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    
    # encode plotly graphs    in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(figures)]
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    print(model.predict([query]))

    # use model to predict classification for query
    classification_label = model.predict([query])[0]

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result={"result": classification_label}
    )