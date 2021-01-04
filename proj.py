from tkinter import *
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt   #doesnot write(%matplotlib inline ) when using SPYDER Editor

expr =""

iris = load_iris()

#input and output
X = iris.data #input
Y = iris.target #output
#print(X.shape) #(150,4)
#print(Y.shape) #(150,)

#split the data for training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 11)
'''print(X_train.shape) #(120,4)
print(X_test.shape) #(30,4)
print(Y_train.shape) #(120,)
print(Y_test.shape) #(30,)'''
#print(X_test) 

w = Tk()
w.geometry("690x470")
v1 = StringVar()
v2 = StringVar()
v3 = StringVar()
v4 = StringVar()

#Create a model KNN
# KNN-Knearest neighbour Algorithm

global K
K = KNeighborsClassifier(n_neighbors = 5)

#train the model
K.fit(X_train,Y_train)

#test the model
Y_pred_knn = K.predict(X_test)

#find Accuracy
global acc_knn
acc_knn = accuracy_score(Y_test,Y_pred_knn)
acc_knn = round(acc_knn*100,2)
#-------------------------------------------------

#Create a Logistic Regression
L = LogisticRegression()

#train the model
L.fit(X_train,Y_train)

#test the model
Y_pred_lg = L.predict(X_test)

#find Accuracy
global acc_lg
acc_lg = accuracy_score(Y_test,Y_pred_lg)
acc_lg = round(acc_lg*100,2)
#-------------------------------------------------

#Create a Decision tree classifier
D = DecisionTreeClassifier()

#train the model
D.fit(X_train,Y_train)

#test the model
Y_pred_dt = D.predict(X_test)

#find Accuracy
global acc_dt
acc_dt = accuracy_score(Y_test,Y_pred_dt)
acc_dt = round(acc_dt*100,2)
#----------------------------------------------------
#Create a Naive Bayes Algorithm
N = GaussianNB()

#train the model
N.fit(X_train,Y_train)

#test the model
Y_pred_nb = N.predict(X_test)

#find Accuracy
global acc_nb
acc_nb = accuracy_score(Y_test,Y_pred_nb)
acc_nb = round(acc_nb*100,2)
#------------------------------------------------------

#Definatin of Calling Functions
def knn():
    print("accuracy in K-nearest neighbors Model is",acc_knn)

def lg():
   print("accuracy in Logistic Regression Model is",acc_lg)

def dt():
   print("accuracy in Decision Tree Model is",acc_dt)

def nb():
    print("accuracy in Naive Bayes Model is",acc_nb)

def compare():
    models = ['KNN','LG','DT','NB']
    accuracy = [acc_knn,acc_lg,acc_dt,acc_nb]
    plt.bar(models,accuracy,color = ['green','blue','violet','red'])
    plt.xlabel("MODELS")
    plt.ylabel("ACCURACY")
    plt.show()

def submit():
    v1 = int()
    v2 = int()
    v3 = int()
    v4 = int()
    print("Prediction of given new flower data is ",K.predict([[v1,v2,v3,v4]]))
    
    
def reset():
    global expr
    expr = ""
    v1.set(expr)
    v2.set(expr)
    v3.set(expr)
    v4.set(expr)
    


#Design All Componenets
L1 = Label(w,bg = "green", text = "IRIS FLOWER PREDICTION", font = ("aerial",38,"bold","underline"))
B1 = Button(w,text = "KNN", bg = "grey",font = ("Times New Roman",17,"bold"),command = knn)
B2 = Button(w,text = "LG", bg = "grey", font = ("Times New Roman",17,"bold"),command = lg)
B3 = Button(w,text = "DT", bg = "grey", font = ("Times New Roman",17,"bold"),command = dt)
B4 = Button(w,text = "NB", bg = "grey", font = ("Times New Roman",17,"bold"),command = nb)
B5 = Button(w,text = "Compare", justify = "center", bg = "blue", font = ("Times New Roman",17,"bold"),command = compare)
L2 = Label(w,bg = "green", text = "Enter Data of New Flower", font = ("aerial",40,"bold"))
L3 = Label(w,bg = "yellow", text = "SL", font = ("Times New Roman",17,"bold"))
L4 = Label(w,bg = "yellow", text = "SW", font = ("Times New Roman",17,"bold"))
L5 = Label(w,bg = "yellow", text = "PL", font = ("Times New Roman",17,"bold"))
L6 = Label(w,bg = "yellow", text = "PW", font = ("Times New Roman",17,"bold"))
E1 = Entry(w,textvariable = v1, justify = "right", bg = "cyan", font = ("Times New Roman",17,"bold"))
E2 = Entry(w,textvariable = v2, justify = "right", bg = "cyan", font = ("Times New Roman",17,"bold"))
E3 = Entry(w,textvariable = v3, justify = "right", bg = "cyan", font = ("Times New Roman",17,"bold"))
E4 = Entry(w,textvariable = v4, justify = "right", bg = "cyan", font = ("Times New Roman",17,"bold"))
B6 = Button(w,text = "Submit", bg = "blue", font = ("Times New Roman",17,"bold"),command = submit)
B7 = Button(w,text = "Reset", bg = "blue", font = ("Times New Roman",17,"bold"),command = reset)

#place all Components at proper position
#row 1
L1.grid(row = 1, column = 1, columnspan = 4, padx = 10, pady  = 10)

#row 2
B1.grid(row = 2, column = 1, padx = 10, pady  = 10)
B2.grid(row = 2, column = 2, padx = 10, pady  = 10)
B3.grid(row = 2, column = 3, padx = 10, pady  = 10)
B4.grid(row = 2, column = 4, padx = 10, pady  = 10)

#row 3
B5.grid(row = 3,column = 1,columnspan =4, padx = 10, pady  = 10)

#row 4
L2.grid(row = 4, column = 1, columnspan = 4, padx = 10, pady  = 10)

#row 5
L3.grid(row = 5, column = 1, padx = 10, pady  = 10)
E1.grid(row = 5, column = 2, padx = 10, pady  = 10)
L4.grid(row = 5, column = 3, padx = 10, pady  = 10)
E2.grid(row = 5, column = 4, padx = 10, pady  = 10)

#row 6
L5.grid(row = 6, column = 1, padx = 10, pady  = 10)
E3.grid(row = 6, column = 2, padx = 10, pady  = 10)
L6.grid(row = 6, column = 3, padx = 10, pady  = 10)
E4.grid(row = 6, column = 4, padx = 10, pady  = 10)

#row 7
B6.grid(row = 10, column = 1,columnspan = 3)
B7.grid(row = 10, column = 3,columnspan = 3)


w.mainloop()
                                                                   
