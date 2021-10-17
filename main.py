from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home():
    prediction = 0
    if request.method == 'POST':
        model = pickle.load(open('clf_model.pkl', 'rb'))
        user_input = []
        for item in request.form.items():
            item = float(item[1])
            user_input.append(item)
        prediction = model.predict([user_input])
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)