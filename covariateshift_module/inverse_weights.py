
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pdb
from sklearn.preprocessing import PolynomialFeatures
def create_model(model:str):
    if model=='linear':
        return LinearRegression()
    elif model=='binary':
        return LogisticRegression()
    else:
        raise ValueError('Not implemented')

class InverseWeight:

    def __init__(self, model:str='linear'):

        self.model = model 
        self.fitting = False

    def __str__(self):
            
        return "InverseWeight"
    
    def fit_model(self, X_train, y_train, X_test, y_test):
        '''
        This method used to compare the result from train and test
        From that we can adjust the weights of training imperically
        '''
        self.forward_model = create_model(self.model)
        ## fit training data
        self.forward_model.fit(X_train, y_train)
        self.backward_model = create_model(self.model)
        ## fit test data
        self.backward_model.fit(X_test, y_test)
        self.fitting = True
       
        pass
    
    def compute_weights(self, X_train, y_train):
        assert self.fitting, "Need to fit models first"
        ## predict the training data
        y_train_pred = self.forward_model.predict(X_train)
        ## predict the testing data
        # y_test_pred = model.predict(X_test)
        ## calculate the error
        train_error = (y_train - y_train_pred)**2
        # test_error = (y_test - y_test_pred)**2
         ## predict the training data
        test_y_train_pred = self.backward_model.predict(X_train)
        ## predict the testing data
        test_train_error = (y_train - test_y_train_pred)**2

        ## if we assume the prediction will follow normal distirbution with the same variance
        ## the weights can be calculated by mse error
        ## the weights probably should be calculated by the ratio 
        # pdb.set_trace()
        # ratio = np.exp(-test_train_error)/np.exp(-train_error)
        return train_error/test_train_error
    

    def compare_the_loss(self, X_train, y_train, X_test, y_test):
        assert self.fitting, "Need to fit models first"
        
        ## first we need to compute the weights
        weights = self.compute_weights(X_train, y_train).flatten()
        ## compute the loss in test
        y_test_pred = self.forward_model.predict(X_test)
        test_error = mean_squared_error(y_test, y_test_pred)
        ## init new model
        new_model = create_model(self.model)
        ## fit the new model with the weights
        new_model.fit(X_train, y_train, sample_weight=weights)
        ## predict the test data
        y_test_pred = new_model.predict(X_test)
        ## calculate the error
        test_error_2 = mean_squared_error(y_test, y_test_pred)
        return test_error, test_error_2
        


from covariateshift_module.uLSIF import  KLIEP, uLSIF
def experiment():
    x_nu = 10 + 2*np.random.normal(size=(10000,1))
    x_de = 3 + 1 * np.random.normal(size=(4000,1))
    

    def func(x):
        poly = PolynomialFeatures(8)
        x= poly.fit_transform(x)
        weights = np.linspace(1,3,9).reshape(-1,1)
        return x @  weights + 5
    y_nu = func(x_nu)
    y_de = func(x_de)

    inverse_model = InverseWeight()
    inverse_model.fit_model(x_de,y_de,
                            x_nu, y_nu)

    model = KLIEP(num_basis=20,basis_func='radial', learning_rate=1e-6,a_tol=1e-6,cv=5)
    model.fit(x_de,x_nu)

    model_u = uLSIF(alpha=0.8,num_basis=200,k_fold=5,basis_func='radial')
    model_u.fit(x_de, x_nu)
    error1,error2 = inverse_model.compare_the_loss(x_de,y_de,
                            x_nu, y_nu)
    
    print(f"Error1: {error1:.2f} Error2: {error2:.2f}")
    fig,ax = plt.subplots()
    x = np.linspace(3,7.,40000)
    y = func(x.reshape(-1,1))
    
    estimate_weights = inverse_model.compute_weights(x.reshape(-1,1), y )
    estimate_weights = estimate_weights/np.mean(estimate_weights)
    # print(f"weights: {estimate_weights}")
    new_weights =  model.compute_weights(x.reshape(-1,1))
    u_weights =  model_u.compute_weights(x.reshape(-1,1))
    ax.plot(x,norm.pdf(x, loc=10, scale=2),label='numerator')
    ax.plot(x,norm.pdf(x, loc=3, scale=1),label='denominator')
    ax.plot(x,norm.pdf(x, loc=10, scale=2)/norm.pdf(x, loc=3, scale=1),label='weight')
    ax.plot(x,estimate_weights.flatten(),label='est weight')
    ax.plot(x,new_weights.flatten(),label='KLIEP est weight')
    ax.plot(x,u_weights.flatten(),label='uLSIF est weight')
    fig.legend()
    fig.savefig('covariateshift_code/estimate_weights.png')

# experiment()

