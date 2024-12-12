from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pickle
import json
import torch
import helper_functions

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import pickle

app = FastAPI()

# Load the model from the pickle file
with open('lstm_model.pkl', 'rb') as f:
    model = pickle.load(f)
    model.to('cpu')

# HTML for the form and prediction display
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 0;
        }
        .container {
            width: 50%;
            margin: 100px auto;
            background-color: #fff;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            border-radius: 8px;
        }
        h2 {
            text-align: center;
            color: #333;
        }
        textarea {
            width: 100%;
            height: 100px;
            padding: 10px;
            margin: 10px 0;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        input[type="submit"] {
            width: 100%;
            padding: 10px;
            background-color: #5cb85c;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        input[type="submit"]:hover {
            background-color: #4cae4c;
        }
        .predictions {
            margin-top: 20px;
            padding: 15px;
            background-color: #e9ecef;
            border-radius: 4px;
        }
        .predictions h3 {
            margin: 0;
            color: #555;
        }
        .prediction-result {
            margin-top: 10px;
            color: #007bff;
            font-weight: bold;
        }
        .prediction-list {
            list-style-type: none;
            padding: 0;
        }
        .prediction-list li {
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Enter Text for Prediction</h2>
        <form action="/predict/" method="post">
            <textarea name="input_text" placeholder="Type something here..."></textarea><br><br>
            <input type="submit" value="Submit">
        </form>
        <div class="predictions">
            <h3>Next Word Prediction:</h3>
            <ul class="prediction-list">
                {{ prediction_list }}
            </ul>
        </div>
        <div class="predictions">
            <h3>Next Few Word Predictions(3 Words):</h3>
            <ul class="prediction-list">
                {{ next_predictions_list }}
            </ul>
        </div>
    </div>
</body>
</html>
"""

def p(model,list_of_words):
    DICT=json.load( open( "vector_dict.json" ) )
    inverse_dict=json.load(open('inverse_dict.json'))
    gg=[DICT[i] for i in list_of_words]
    z=len(gg)
    if z<5:
        while(len(gg)<5):
            gg.append(0)
    else:
        gg=gg[-5:]
    gg=torch.tensor(gg).unsqueeze(0)
    output=model(gg).squeeze(0).tolist()
    
    answer =inverse_dict[str(output.index(max(output)))]  # for next word

    # for next few words
    list_of_words.append(answer)
    hold=list_of_words
    for i in range(3):
        gg=[DICT[i] for i in hold]
        z=len(gg)
        if z<5:
            while(len(gg)<5):
                gg.append(0)
        else:
            gg=gg[-5:]
        gg=torch.tensor(gg).unsqueeze(0)
        output=model(gg).squeeze(0).tolist()
        hold.append(inverse_dict[str(output.index(max(output)))])


    return [answer],hold[-3:]


@app.get("/", response_class=HTMLResponse)
async def index():
    # Render the HTML form
    return html_content.replace("{{ prediction_list }}", "").replace("{{ next_predictions_list }}", "")

@app.post("/predict/", response_class=HTMLResponse)
async def predict(input_text: str = Form(...)):
    # Make the prediction using the model
    prediction,next_predictions = p(model,input_text.split())
    
    # Assuming predictions is a list of words, e.g., ["word1", "word2", "word3"]
    prediction_list_html = "".join([f"<li>{pred}</li>" for pred in prediction])

    # Assuming you also want to display the next few predictions
    # next_predictions = model.predict_next([input_text])  # Adjust based on how you get next predictions
    next_predictions_list_html = "".join([f"<li>{pred}</li>" for pred in next_predictions])

    # Render the form and display the prediction results
    return html_content.replace("{{ prediction_list }}", prediction_list_html).replace("{{ next_predictions_list }}", next_predictions_list_html)



