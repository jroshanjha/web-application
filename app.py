from flask import Flask , jsonify, request, render_template
import pickle 
import numpy as np

app = Flask(__name__)

# Load the model 
with open('model.pkl', 'rb') as model_file: 
    model = pickle.load(model_file)

@app.route('/',methods=['GET'])
def index():
    #return 'This is flask web application'
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input data
    data = [float(x) for x in request.form.values()]
    # Make prediction
    prediction = model.predict([data])
    output = "Iris-setosa" if int(prediction[0])==0 else "Iris-versicolor" if int(prediction[0])==1 else "Iris-virginica"
    #return render_template('index.html', prediction_text=f'Predicted Value: {output}')
    return render_template('result.html', prediction=output)

@app.route('/api/predict', methods=['POST']) 
def predict_api(): 
    if request.method=='POST':
        data = request.get_json(force=True)
        # for i in data.keys():
        #     data[i] = float(data[i])
        # Ensures numpy array is of type float32
        features = np.array(data['data']).astype(np.float32)
        #prediction = model.predict([np.array(data['data'])]) 
        prediction = model.predict([features])
        output = "Iris-setosa" if int(prediction[0])==0 else "Iris-versicolor" if int(prediction[0])==1 else "Iris-virginica"
        return jsonify({'prediction': output}) , 201
    return jsonify({'prediction': 'error'}),404

if __name__=='__main__':
    app.run(debug=True,port='9200')