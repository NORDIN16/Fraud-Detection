import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.plk', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]

    # Perform the operations to calculate errorBalanceOrig and errorBalanceDest
    newbalanceOrig = float(final_features[0][6])
    amount = float(final_features[0][0])
    oldbalanceOrg = float(final_features[0][3])
    oldbalanceDest = float(final_features[0][4])
    newbalanceDest = float(final_features[0][5])
    step = final_features[0][2]
    type = final_features[0][1]
    
    errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
    errorBalanceDest = oldbalanceDest + amount - newbalanceDest

    # Create a new list with the updated features
    updated_features = [step, type,	amount,	oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, errorBalanceOrig, errorBalanceDest]

    # Assuming updated_features is a 1D array
    updated_features = np.array(updated_features)

    # Reshape the array to have a shape of (1, num_features)
    updated_features = updated_features.reshape(1, -1)

    # Perform prediction
    prediction = model.predict(updated_features)


    output = round(prediction[0], 2)

    if (output == 0) :
        return render_template('index.html', prediction_text='Not Fraud ')
    else : 
        return render_template('index.html', prediction_text='Is Fraud' )

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)