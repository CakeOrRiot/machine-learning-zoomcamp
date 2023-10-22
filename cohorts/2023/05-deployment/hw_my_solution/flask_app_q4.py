from flask import Flask, request, jsonify
import pickle

app = Flask('predict') # give an identity to your web service


with open('model1.bin', 'rb') as f_model:
    model = pickle.load(f_model)
with open("dv.bin", "rb") as f_dv: 
    dv = pickle.load(f_dv)

@app.route('/predict', methods=['POST']) # use decorator to add Flask's functionality to our function
def predict():
    
    client = request.get_json()

    proba = model.predict_proba(dv.transform([client]))[:,1][0]
    
    result = {"proba" : float(proba), "class": bool(proba>=0.5)}
    return jsonify(result)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
    