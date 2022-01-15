from flask import Flask,request,render_template
import pickle
from flask_cors import cross_origin
app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
@cross_origin()
def index():
    return render_template('/index.html')

@app.route('/predict',methods=['GET','POST'])
@cross_origin()
def predict():
    if request.method=='POST':
        try:
            rm = float(request.form["rm"])
            ptratio = float(request.form["ptratio"])
            lstat = float(request.form["lstat"])
            indus = float(request.form["indus"])
            # df = pd.DataFrame([rm, lstat, ptratio, indus])
            filename = 'MLAssign_RandomForest.pickle'
            loadedmodel = pickle.load(open(filename, "rb"))
            predicted_price = loadedmodel.predict([[rm, lstat, ptratio, indus]])
            print('prediction is', predicted_price)
            # return render_template('results.html', prediction=round(100 * predicted_price[0]))
            return render_template('result.html', prediction=predicted_price[0])
        except Exception as e:
            print('Exception is ',e)
            return('Something wrong')
        else:
            return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
