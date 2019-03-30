import json
import plotly
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def f1_micro_average(y_test,y_pred):
    '''
    Calculates s.c. micro average f1
    Input: y_test, y_pred - arrays with test and prediction values
    Output: micro average f1
    '''
    
    TN = []
    FP = []
    FN = []
    for i in range(y_pred.shape[1]):
        TN.append(confusion_matrix(np.array(y_test)[:,i],y_pred[:,i])[1,1])
        FP.append(confusion_matrix(np.array(y_test)[:,i],y_pred[:,i])[1,0])
        FN.append(confusion_matrix(np.array(y_test)[:,i],y_pred[:,i])[0,1])
    precision = np.sum(TN) / (np.sum(TN) + np.sum(FN))
    recall = np.sum(TN) / (np.sum(TN) + np.sum(FP))
    
    return hmean([precision,recall])

# load data
engine = create_engine('sqlite:///../data/twitter_messages.db')
df = pd.read_sql_table('message', engine)

# load model
model = joblib.load("../models/model.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    data_plot = df.groupby(['genre','related'])['message'].count().reset_index()
    data_plot['name'] = data_plot['genre'] + '_' + data_plot['related'].astype(str)

    ratio_direct = []
    ratio_news = []
    ratio_social = []
    
    new_df = df.iloc[:,4:]
    
    for i in new_df.columns:
        ratio_direct.append(df.loc[df['genre'] == 'direct',i].sum() / df.loc[df['genre'] == 'direct'].shape[0])
        ratio_news.append(df.loc[df['genre'] == 'news',i].sum() / df.loc[df['genre'] == 'news'].shape[0])
        ratio_social.append(df.loc[df['genre'] == 'social',i].sum() / df.loc[df['genre'] == 'social'].shape[0])

    skew_data = pd.DataFrame({
        'group':new_df.columns,
        'ratio_direct':ratio_direct,
        'ratio_news':ratio_news,
        'ratio_social':ratio_social
    })
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=skew_data['group'],
                    y=skew_data['ratio_direct']
                )
            ],

            'layout': {
                'title': 'Proportion of "1" in the groups for genre "direct"',
                'yaxis': {
                    'title': "Ratio of 1 in the group"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },
   
        {
            'data': [
                Bar(
                    x=skew_data['group'],
                    y=skew_data['ratio_news']
                )
            ],

            'layout': {
                'title': 'Proportion of "1" in the groups for genre "news"',
                'yaxis': {
                    'title': "Ratio of 1 in the group"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },
        {
            'data': [
                Bar(
                    x=skew_data['group'],
                    y=skew_data['ratio_social']
                )
            ],

            'layout': {
                'title': 'Proportion of "1" in the groups for genre "social"',
                'yaxis': {
                    'title': "Ratio of 1 in the group"
                },
                'xaxis': {
                    'title': ""
                }
            }
        }
    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()