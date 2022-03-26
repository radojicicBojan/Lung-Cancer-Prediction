# pylint: disable=no-member
import pandas as pd
import scipy.optimize as op 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import mean
from numpy import absolute
from numpy import sqrt
import keyboard 

# Read CSV file
df = pd.read_csv(r'./survey_lung_cancer.csv')

# Dictionary: YES  == 1 || NO     == 0
#             MALE == 1 || FEMALE == 0  
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 0})

"""
# Determining interdependence using correlation: 
print(np.corrcoef(df['GENDER'], df['LUNG_CANCER']))
print(np.corrcoef(df['AGE'], df['LUNG_CANCER']))
print(np.corrcoef(df['SMOKING'], df['LUNG_CANCER']))
print(np.corrcoef(df['YELLOW_FINGERS'], df['LUNG_CANCER']))
print(np.corrcoef(df['ANXIETY'], df['LUNG_CANCER']))
print(np.corrcoef(df['PEER_PRESSURE'], df['LUNG_CANCER']))
print(np.corrcoef(df['CHRONIC_DISEASE'], df['LUNG_CANCER']))
print(np.corrcoef(df['FATIGUE'], df['LUNG_CANCER']))
print(np.corrcoef(df['ALLERGY'], df['LUNG_CANCER']))           
print(np.corrcoef(df['WHEEZING'], df['LUNG_CANCER']))
print(np.corrcoef(df['ALCOHOL_CONSUMING'], df['LUNG_CANCER']))
print(np.corrcoef(df['COUGHING'], df['LUNG_CANCER']))
print(np.corrcoef(df['SHORTNESS_OF_BREATH'], df['LUNG_CANCER']))
print(np.corrcoef(df['SWALLOWING_DIFFICULTY'], df['LUNG_CANCER']))
print(np.corrcoef(df['CHEST_PAIN'], df['LUNG_CANCER']))

sns.heatmap(df.corr())
# Nema redudantnih podataka (Podataka sa velikom koorelacijom)
"""

# Making X and Y
y = np.asarray(df.LUNG_CANCER.values)
K = df.iloc[: , : 15]
X = np.asarray(K.T)


# The Logistic Function
def sigmoidFunction(z): 
    return 1 / (1 + np.exp(-z))

# Logistic Regression Cost Function
def costFunction(theta, X, y):
    
    # number of training examples
    m,n = X.shape  
        
    # Calculate the Cost J
    J = -1 / m * (np.sum(y * np.log(sigmoidFunction(X.dot(theta))) + (1 - y) * np.log(1 - sigmoidFunction( X.dot(theta)))))
    return J


# Calculate the accuracy
def CalcAccuracy(theta, X):
    p = sigmoidFunction(X.dot(theta)) >= 0.5
    return p

# The Gradient Function
def calcGradient(theta, X, y):
    
    # number of training examples
    m,n = X.shape    
    
    gradient = 1/m * (X.T).dot(sigmoidFunction(X.dot(theta)) - y)
    
    return gradient    


def FeatureScalingNormalizationMultipleVariables(X):
    
  
    X_norm = X 

    # mean
    mi = np.zeros(X.shape[1]) 
    
    # standard deviation
    sigma = np.zeros(X.shape[1]) 

    mi = np.vstack((X[0].mean(), \
                    X[1].mean(), \
                    X[2].mean(), \
                    X[3].mean(), \
                    X[4].mean(), \
                    X[5].mean(), \
                    X[6].mean(), \
                    X[7].mean(), \
                    X[8].mean(), \
                    X[9].mean(), \
                    X[10].mean(), \
                    X[11].mean(), \
                    X[12].mean(), \
                    X[13].mean(), \
                    X[14].mean()))
   

    sigma = np.vstack((X[0].std(ddof=1),\
                       X[1].std(ddof=1),\
                       X[2].std(ddof=1),\
                       X[3].std(ddof=1),\
                       X[4].std(ddof=1),\
                       X[5].std(ddof=1),\
                       X[6].std(ddof=1),\
                       X[7].std(ddof=1),\
                       X[8].std(ddof=1),\
                       X[9].std(ddof=1),\
                       X[10].std(ddof=1),\
                       X[11].std(ddof=1),\
                       X[12].std(ddof=1),\
                       X[13].std(ddof=1),\
                       X[14].std(ddof=1)))     

    m = X.shape[1] 
    
    mi_matrix = np.multiply(np.ones(m), mi).T 
    
    sigma_matrix = np.multiply(np.ones(m), sigma).T
    
    X_norm = np.subtract(X, mi).T
    X_norm = X_norm /sigma.T
    
    return [X_norm, mi, sigma]


featuresNormalizeresults = FeatureScalingNormalizationMultipleVariables(X)

# normalized X matrix
X = np.asarray(featuresNormalizeresults[0]).T

# mean
mi = featuresNormalizeresults[1]

sigma = featuresNormalizeresults[2]

# number of training examples
m = len(y) 

# number of features
n = len(X)

X = np.vstack((np.ones(m), X)).T

# Find the optimal theta
m , n = X.shape
initial_theta = np.zeros(n)
Result = op.minimize(fun = costFunction, 
                                 x0 = initial_theta, 
                                 args = (X, y),
                                 method = 'Newton-CG',
                                 jac = calcGradient)
theta = Result.x
message = Result.message

x_input = []
print("Procena rizika od dobijanja raka pluća.")
print("Molimo popunite sledeći upitnik:")
print("Vaš pol: (M - Muški F - Ženski)")
while True:
    if keyboard.is_pressed("M"):
        x_input.append(0)
        print("Muško")
        break
    elif keyboard.is_pressed("F"):
        x_input.append(1)
        print("Žensko")
        break

print("Vaše godine: ")
x_input.append(input()) 
print("Da li ste aktivni pušač cigareta: (2 - DA || 1 - NE)")
x_input.append(input()) 
print("Da li imate žute fleke na prstima: (2 - DA || 1 - NE)")
x_input.append(input()) 
print("Da li ste anksiozni: (2 - DA || 1 - NE)")
x_input.append(input()) 
print("Da li imate povišeni krvni pritisak: (2 - DA || 1 - NE)")
x_input.append(input()) 
print("Da li imate neko hronično oboljenje: (2 - DA || 1 - NE)")
x_input.append(input()) 
print("Da li ste hronično umorni: (2 - DA || 1 - NE)")
x_input.append(input()) 
print("Da li imate alergiju: (2 - DA || 1 - NE)")
x_input.append(input()) 
print("Da li teško dišete: (2 - DA || 1 - NE)")
x_input.append(input()) 
print("Da li konzumirate alkohol: (2 - DA || 1 - NE)")
x_input.append(input()) 
print("Da li teško gutate: (2 - DA || 1 - NE)")
x_input.append(input()) 
print("Da li često kašljete: (2 - DA || 1 - NE)")
x_input.append(input()) 
print("Da li imate kratak dah: (2 - DA || 1 - NE)")
x_input.append(input()) 
print("Da li imate bol u plućima: (2 - DA || 1 - NE)")
x_input.append(input()) 


query = np.asfarray(x_input, float)

# Scale and Normalize the query
query_Normalized = np.asarray([1, ((query[0]-float(mi[0]))/float(sigma[0])),\
                               ((query[1]-float(mi[1]))/float(sigma[1])),\
                               ((query[2]-float(mi[2]))/float(sigma[2])),\
                               ((query[3]-float(mi[3]))/float(sigma[3])),\
                               ((query[4]-float(mi[4]))/float(sigma[4])),\
                               ((query[5]-float(mi[5]))/float(sigma[5])),\
                               ((query[6]-float(mi[6]))/float(sigma[6])),\
                               ((query[7]-float(mi[7]))/float(sigma[7])),\
                               ((query[8]-float(mi[8]))/float(sigma[8])),\
                               ((query[9]-float(mi[9]))/float(sigma[9])),\
                               ((query[10]-float(mi[10]))/float(sigma[10])),\
                               ((query[11]-float(mi[11]))/float(sigma[11])),\
                               ((query[12]-float(mi[12]))/float(sigma[12])),\
                               ((query[13]-float(mi[13]))/float(sigma[13])),\
                               ((query[14]-float(mi[14]))/float(sigma[14]))])


# Calculate the prediction using the Logistic Function
prediction = sigmoidFunction(query_Normalized.dot(theta))

# Calculate accuracy
p = CalcAccuracy(theta, X)
p = (p == y) * 100

# Print the output for programmer
prediction_print = round(prediction, 2)*100
print("Message: "+str(message))
print("Train Accuracy: "+str(p.mean()))

y_pred = prediction

# Messages for client
print("Rizik za dobijanje raka je: "+str(prediction_print)+"%")
if y_pred < 0.3 :
    print("Imate jako nizak rizik od dobijanja raka pluća.")
if y_pred >= 0.3 and y_pred < 0.4 :
    print("Imate nizak rizik od dobijanja raka pluća. Pokušajte da unapredite Vaš način života")
if y_pred >= 0.4 and y_pred < 0.7 :
    print("Imate srednji rizik od dobijanja raka pluća. Pokušajte da unapredite Vaš način života")
if y_pred >= 0.7 and y_pred < 0.9 :
    print("Imate visok rizik od dobijanja raka pluća. Pokušajte da unapredite Vaš način života što pre")
if y_pred >= 0.9 :
    print("Imate ekstremno visok rizik od dobijanja raka pluća. Radite na što zdravijem načinu života što pre")
    