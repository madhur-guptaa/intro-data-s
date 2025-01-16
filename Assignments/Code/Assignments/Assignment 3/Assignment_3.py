#%% md
# 
# # Assignment 3 for Course 1MS041
# Make sure you pass the `# ... Test` cells and
#  submit your solution notebook in the corresponding assignment on the course website. You can submit multiple times before the deadline and your highest score will be used.
#%% md
# ---
# ## Assignment 3, PROBLEM 1
# Maximum Points = 8
#%% md
# 
# Download the updated data folder from the course github website or just download directly the file [https://github.com/datascience-intro/1MS041-2024/blob/main/notebooks/data/smhi.csv](https://github.com/datascience-intro/1MS041-2024/blob/main/notebooks/data/smhi.csv) from the github website and put it inside your data folder, i.e. you want the path `data/smhi.csv`. The data was aquired from SMHI (Swedish Meteorological and Hydrological Institute) and constitutes per hour measurements of wind in the Uppsala Aut station. The data consists of windspeed and direction. Your goal is to load the data and work with it a bit. The code you produce should load the file as it is, please do not alter the file as the autograder will only have access to the original file.
# 
# The file information is in Swedish so you need to use some translation service, for instance `Google translate` or ChatGPT.
# 
# 1. [2p] Load the file, for instance using the `csv` package. Put the wind-direction as a numpy array and the wind-speed as another numpy array.
# 2. [2p] Use the wind-direction which is an angle in degrees and convert it into a point on the unit circle. Store the `x_coordinate` as one array and the `y_coordinate` as another. From these coordinates, construct the wind-velocity vector.
# 3. [2p] Calculate the average wind velocity and convert it back to direction and compare it to just taking average of the wind direction as given in the data-file.
# 4. [2p] The wind velocity is a $2$-dimensional random variable, calculate the empirical covariance matrix which should be a numpy array of shape (2,2).
# 
# For you to wonder about, is it more likely for you to have headwind or not when going to the university in the morning.
#%%
import csv

file_path = 'data/smhi.csv'

wind_data_headers = []
wind_data = []

# Read the file
with open(file_path, mode='r', encoding='utf-8') as file:
    csv_reader = csv.reader(file, delimiter=';')

    i = 0
    for row in csv_reader:
        if len(row) <= 5:
            continue

        if row[0] == 'Datum':
            wind_data_headers.append(row[0:6])
            continue

        wind_data.append(row[0:6])

#%%
import pandas as pd
import numpy as np

df = pd.DataFrame(data=wind_data, columns=wind_data_headers)
df.drop(df.columns[[3, 5]], axis=1, inplace=True)
df['Vindriktning'] = df['Vindriktning'].astype(float)
df['Vindhastighet'] = df['Vindhastighet'].astype(float)
#%%

problem1_wind_direction = np.array(df['Vindriktning'])
problem1_wind_speed = np.array(df['Vindhastighet'])
#%%
df['wind-x-coordinate'] = np.radians(df['Vindriktning'])
df['wind-y-coordinate'] = np.sin(df['wind-x-coordinate'])
df['wind-x-coordinate'] = np.cos(df['wind-x-coordinate'])
df['velocity-x-coordinate'] = df.apply(lambda row: (row['Vindhastighet'] * row['wind-x-coordinate']), axis=1)
df['velocity-y-coordinate'] = df.apply(lambda row: (row['Vindhastighet'] * row['wind-y-coordinate']), axis=1)
#%%

# The wind direction is given as a compass direction in degrees (0-360)
# convert it to x and y coordinates using the standard mathematical convention
problem1_wind_direction_x_coordinate = np.array(df['wind-x-coordinate'])
problem1_wind_direction_y_coordinate = np.array(df['wind-y-coordinate'])


problem1_wind_velocity_x_coordinate = np.array(df['velocity-x-coordinate'])
problem1_wind_velocity_y_coordinate = np.array(df['velocity-y-coordinate'])
#%%

#%%

# Put the average wind velocity x and y coordinates here in these variables
problem1_average_wind_velocity_x_coordinate = np.mean(problem1_wind_velocity_x_coordinate)
problem1_average_wind_velocity_y_coordinate = np.mean(problem1_wind_velocity_y_coordinate)

# First calculate the angle of the average wind velocity vector in degrees
problem1_average_wind_velocity_angle_degrees = np.degrees(np.tanh(
    problem1_average_wind_velocity_y_coordinate / problem1_average_wind_velocity_x_coordinate))
# Then calculate the average angle of the wind direction in degrees (using the wind direction in the data)
problem1_average_wind_direction_angle_degrees = np.mean(np.array(df['Vindriktning']))

# Finally, are they the same? Answer as a boolean value (True or False)
problem1_same_angle = False
#%%
wind_velocity = np.column_stack([problem1_wind_velocity_x_coordinate, problem1_wind_velocity_y_coordinate])
cov_matrix = np.cov(wind_velocity, rowvar=False)
#%%

problem1_wind_velocity_covariance_matrix = cov_matrix
#%% md
# ---
# ## Assignment 3, PROBLEM 2
# Maximum Points = 8
#%% md
# 
# For this problem you will need the [pandas](https://pandas.pydata.org/) package and the [sklearn](https://scikit-learn.org/stable/) package. Inside the `data` folder from the course website you will find a file called `indoor_train.csv`, this file includes a bunch of positions in (X,Y,Z) and also a location number. The idea is to assign a room number (Location) to the coordinates (X,Y,Z).
# 
# 1. [2p] Take the data in the file `indoor_train.csv` and load it using pandas into a dataframe `df_train`
# 2. [3p] From this dataframe `df_train`, create two numpy arrays, one `Xtrain` and `Ytrain`, they should have sizes `(1154,3)` and `(1154,)` respectively. Their `dtype` should be `float64` and `int64` respectively.
# 3. [3p] Train a Support Vector Classifier, `sklearn.svc.SVC`, on `Xtrain, Ytrain` with `kernel='linear'` and name the trained model `svc_train`.
# 
# To mimic how [kaggle](https://www.kaggle.com/) works, the Autograder has access to a hidden test-set and will test your fitted model.
#%%
df_train = pd.read_csv('data/indoor_train.csv')
#%%

Xtrain = df_train[['Position X', ' Position Y', 'Position Z']].to_numpy()
Ytrain = df_train['Location'].to_numpy()
#%%
from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(Xtrain, Ytrain)
#%%

svc_train = model
#%% md
# ---
# ## Assignment 3, PROBLEM 3
# Maximum Points = 8
#%% md
# 
# Let us build a proportional model ($\mathbb{P}(Y=1 \mid X) = G(\beta_0+\beta \cdot X)$ where $G$ is the logistic function) for the spam vs not spam data. Here we assume that the features are presence vs not presence of a word, let $X_1,X_2,X_3$ denote the presence (1) or absence (0) of the words $("free", "prize", "win")$.
# 
# 1. [2p] Load the file `data/spam.csv` and create two numpy arrays, `problem3_X` which has shape **(n_texts,3)** where each feature in `problem3_X` corresponds to $X_1,X_2,X_3$ from above, `problem3_Y` which has shape **(n_texts,)** and consists of a $1$ if the email is spam and $0$ if it is not. Split this data into a train-calibration-test sets where we have the split $40\%$, $20\%$, $40\%$, put this data in the designated variables in the code cell.
# 
# 2. [2p] Follow the calculation from the lecture notes where we derive the logistic regression and implement the final loss function inside the class `ProportionalSpam`. You can use the `Test` cell to check that it gives the correct value for a test-point.
# 
# 3. [2p] Train the model `problem3_ps` on the training data. The goal is to calibrate the probabilities output from the model. Start by creating a new variable `problem3_X_pred` (shape `(n_samples,1)`) which consists of the predictions of `problem3_ps` on the calibration dataset. Then train a calibration model using `sklearn.tree.DecisionTreeRegressor`, store this trained model in `problem3_calibrator`. Recall that calibration error is the following for a fixed function $f$
# $$
#     \sqrt{\mathbb{E}[|\mathbb{E}[Y \mid f(X)] - f(X)|^2]}.
# $$
# 
# 4. [2p] Use the trained model `problem3_ps` and the calibrator `problem3_calibrator` to make final predictions on the testing data, store the prediction in `problem3_final_predictions`. 
#%%

problem3_X = XXX
problem3_Y = XXX

problem3_X_train = XXX
problem3_X_calib = XXX
problem3_X_test = XXX

problem3_Y_train = XXX
problem3_Y_calib = XXX
problem3_Y_test = XXX

print(problem3_X_train.shape,problem3_X_calib.shape,problem3_X_test.shape,problem3_Y_train.shape,problem3_Y_calib.shape,problem3_Y_test.shape)
#%%


class ProportionalSpam(object):
    def __init__(self):
        self.coeffs = None
        self.result = None
    
    # define the objective/cost/loss function we want to minimise
    def loss(self,X,Y,coeffs):
        
        return XXX

    def fit(self,X,Y):
        import numpy as np
        from scipy import optimize

        #Use the f above together with an optimization method from scipy
        #to find the coefficients of the model
        opt_loss = lambda coeffs: self.loss(X,Y,coeffs)
        initial_arguments = np.zeros(shape=X.shape[1]+1)
        self.result = optimize.minimize(opt_loss, initial_arguments,method='cg')
        self.coeffs = self.result.x
    
    def predict(self,X):
        #Use the trained model to predict Y
        if (self.coeffs is not None):
            G = lambda x: np.exp(x)/(1+np.exp(x))
            return np.round(10*G(np.dot(X,self.coeffs[1:])+self.coeffs[0]))/10 # This rounding is to help you with the calibration
#%%

problem3_ps = XXX

problem3_X_pred = XXX

problem3_calibrator = XXX
#%%

problem3_final_predictions = XXX
#%% md
# ---
# #### Local Test for Assignment 3, PROBLEM 3
# Evaluate cell below to make sure your answer is valid.                             You **should not** modify anything in the cell below when evaluating it to do a local test of                             your solution.
# You may need to include and evaluate code snippets from lecture notebooks in cells above to make the local test work correctly sometimes (see error messages for clues). This is meant to help you become efficient at recalling materials covered in lectures that relate to this problem. Such local tests will generally not be available in the exam.
#%%
try:
    import numpy as np
    test_instance = ProportionalSpam()
    test_loss = test_instance.loss(np.array([[1,0,1],[0,1,1]]),np.array([1,0]),np.array([1.2,0.4,0.3,0.9]))
    assert (np.abs(test_loss-1.2828629432232497) < 1e-6)
    print("Your loss was correct for a test point")
except:
    print("Your loss was not correct on a test point")