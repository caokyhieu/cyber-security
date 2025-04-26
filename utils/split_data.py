from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from metric_learn import NCA
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
from numpy import int64, ndarray
from scipy.stats import beta,bernoulli
import pdb

def read_data_selection_bias(photo_path: str,
                   spectral_path: str,
                   cols: List[str] = ['u_mod','g_mod','r_mod','i_mod','z_cmod'],
                   a:float=2.,b:float=2.,
                    normalize: bool=False
                    ) -> Tuple[Tuple[ndarray, None, ndarray], Tuple[ndarray, None, ndarray]]:
    
    photo_df = pd.read_table(photo_path,skiprows=2,sep=' ')
    spectral_df = pd.read_table(spectral_path,sep=' ')
    ## take the extreme values of spectral data as the range of features
    extreme_dict = {col:(spectral_df[col].min(),spectral_df[col].max()) for col in cols}
    ## filter the photometric data by the extreme values
    length_photo = len(photo_df)
    for col in cols:
        photo_df = photo_df[(photo_df[col]>=extreme_dict[col][0]) & (photo_df[col]<=extreme_dict[col][1])]
        print(f"Photometric data remaining: {len(photo_df)/length_photo * 100:.2f} percent. Filtered by {col}")
    
    ### select train and test depend on the band 'r'
    r_band = spectral_df['r_mod'].values
    ## scale in in  [0.,1.]
    r_band = (r_band - r_band.min())/(r_band.max() - r_band.min())
    ## beta distribution
    rv = beta(a, b)
    ## select train and test based on this beta distribution, the probability is bernoulli distribution with p = beta(x)
    test_data_index_bool = bernoulli.rvs(rv.pdf(r_band)/rv.pdf(r_band).max()).astype(bool)
    # pdb.set_trace()
    test_index = spectral_df.index[test_data_index_bool]
    pseudo_photo = spectral_df.iloc[test_index][cols].to_numpy(dtype=np.float64)
    photo_label = spectral_df.iloc[test_index]['redshift'].to_numpy(dtype=np.float64)
    pseudo_spectro = spectral_df[cols].to_numpy(dtype=np.float64)

    spectro_label = spectral_df['redshift'].to_numpy(dtype=np.float64)
    print("Done")

    n_choice_photo = int(len(pseudo_photo)*0.5)
    n_choice_spectro = int(len(pseudo_spectro)*0.5)
    choice_photo_index = np.random.permutation(len(pseudo_photo))[:n_choice_photo]
    choice_spectro_index = np.random.permutation(len(pseudo_spectro))[:n_choice_spectro]

    if normalize:
        covariate_mean = pseudo_spectro.mean()
        covariate_std = pseudo_spectro.std()
        label_mean = spectro_label.mean()
        label_std = spectro_label.std()
        pseudo_spectro = (pseudo_spectro - covariate_mean)/covariate_std
        spectro_label = (spectro_label - label_mean)/label_std
        pseudo_photo = (pseudo_photo - covariate_mean)/covariate_std
        photo_label = (photo_label - label_mean)/label_std

        return (pseudo_spectro[choice_spectro_index],None,spectro_label[choice_spectro_index]),(pseudo_photo[choice_photo_index],None,photo_label[choice_photo_index] ),(covariate_mean,covariate_std,label_mean,label_std)
    else:
        return (pseudo_spectro[choice_spectro_index],None,spectro_label[choice_spectro_index]),(pseudo_photo[choice_photo_index],None,photo_label[choice_photo_index])
    

class DataSplit:

    def __init__(self,cv=5,random_state=42):
        self.random_state = random_state
        self.cv = cv

    def kfold(self,x,y,x_test,n_samples=300):
        """
        split data into k folds
        """
        nca = NCA(random_state=self.random_state)
        ## sample in train
        train_index = random.sample([i for i in range(0,len(x))],n_samples)
        test_index = random.sample([i for i in range(0,len(x_test))],n_samples)
        X1 = x[train_index]
        X2 = x_test[test_index]
        X = np.concatenate((X1,X2),axis=0)
        label = np.concatenate((np.ones((len(X1),))
                        , np.zeros((len(X2),))),axis=0)
        nca.fit(X, label)
        knn = KNeighborsClassifier(metric=nca.get_metric())
        knn.fit(X, label)
        pred_label = knn.predict(x)
        X_train = x[pred_label==0]
        X_test = x[pred_label==1]
        y_train = y[pred_label==0]
        y_test =  y[pred_label==1]
        ## kfold
       
        train_index = np.random.permutation(len(X_train))
        test_index = np.random.permutation(len(X_test))
        for i in range(self.cv):
            tr_index = list(set(train_index).difference(set(train_index[i::self.cv])))
            te_index = list(set(test_index).difference(set(test_index[i::self.cv])))
            
            yield X_train[tr_index], y_train[tr_index],X_test[te_index],y_test[te_index]
            

