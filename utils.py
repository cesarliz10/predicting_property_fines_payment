# coding: utf-8

import numpy
import pandas as pd
import seaborn as sns
import time
import graphviz

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches


from sklearn import neighbors
from sklearn.tree import export_graphviz
from sklearn.preprocessing import RobustScaler
import category_encoders as ce
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def df_geog_info( verbose=False ):
    '''
    Add  geographical features:  lat, lon,  address, street, zip_code.
    Scaling of lat, long: They were centered around the mean and then normalized using a `RobustScaler`. It removes the median and scales  
    the data according to the quantile range (IQR), making it more robust to outliers. For latitute and longitude we know that almost all 
    the tickets are placed in the metropolitan area so we can  take and  IQR range rather large: (10%, 90%)
    '''
    # dataset with addresses and ticket ids
    dfa = pd.read_csv("addresses.csv")
    dfa.dropna(inplace=True)
    # dataset with lat lon coordinates for different addresses
    dflalo = pd.read_csv("latlons.csv")
    dflalo.dropna(inplace=True)
    # parsing addresses from both dataframes so they have similar format
    address_parser = lambda x: x.split(",")[0].strip()
    dfa["address"] = dfa["address"].apply(address_parser)
    dflalo["address"]  = dflalo["address"].apply(address_parser)    
    
    #joining  data sets by address
    df_geoloc = dfa.set_index("address").join(dflalo.set_index("address"),how="left")
    # An address may be appear on several tickets, so we want a dataframe whose index is the ticket id. by taking the mean only the numerical variables (lat lon) remain 
    df_geoloc = df_geoloc.groupby("ticket_id").agg("mean")

    # we merge the previous dataframe (with index ticket id, and cols (lat,lon) ) with dfa to get again the addresses info
    df_geoloc = df_geoloc.merge(dfa,left_index=True,right_index=True)
    len_df_geol_all = len(df_geoloc)
        #we drop na values just to keep tickets with full info about address, lat and lon.
    df_geoloc.dropna(inplace=True)
    df_geoloc.set_index("ticket_id",inplace=True)
    if verbose:
        print("rows in df with addresses and ticketsID: {}".format(len(dfa)))
        print("rows after joining by address with latlon df: {}".format(len_df_geol_all))
        print("rows with merged info (lat,lon, address) after Nas dropped: {}".format(len(df_geoloc)))
        df_geoloc.head(5)
    return df_geoloc


# In[11]:


def generate_street_zipcode(df1):
    street_parser = lambda x: " ".join( x.split(",")[0].split(" ")[1:]).strip()
    df1["street"] = df1["address"].apply(street_parser)
    df1["zip_code"] = df1["address"].str.findall('\d+').apply( lambda x: x[0])
    return df1


# In[12]:


def encode_categorical(df1,strategy4geog="binary"):
    '''
    The "disposition" variable has only 4 unique values, so we use a **one-hot encoding** strategy to obtain the features.
    For  "zip_code" and "street" variables, the number of unique values is rather large, a one-hot encoding is possible but it generates 
    way too many features. For them a  **binary encoding** is also available. It encodes first each value of the categorical variable into 
    integers and then transforming then into binary. Each digit of the binary expression will be a feature.
    '''
    
    for i in ["compliance","disposition","zip_code","street","address"]:
        if i in df1.columns:
            df1[i] = df1[i].astype("category")
    
    #one-hot encoding for disposition
    df1["disposition"] = df1["disposition"].apply(  lambda x: str(x).split("by")[1].strip().lower()  )
    disposition_dummies = pd.get_dummies(data=df1["disposition"],prefix="responsible_by")
    df1 = pd.concat([ df1.drop(columns="disposition") , disposition_dummies],axis=1)
    
    
    for i in ["zip_code","street","address"]:
        if i in df1.columns:
            if strategy4geog=="binary":
                unique_values = len(df1[i].unique())
                bin_unique_values = bin(unique_values)
                len_bin_unique_values = len(bin_unique_values.split('0b')[1]) +1
                print("Variable {}:\n- {} unique values\n- max. binary representation: {}\n- binary-encoded generated features: {}\n".format(i,unique_values,bin_unique_values,len_bin_unique_values))
                #previously: pip3 install category_encoders
                bin_encoder =  ce.BinaryEncoder(cols=i,return_df=True)
                df1 = bin_encoder.fit_transform(df1)
            if strategy4geog=="onehot":
                disposition_dummies = pd.get_dummies(data=df1["i"],prefix=i)
                df1 = pd.concat([ df1.drop(columns=i) , disposition_dummies],axis=1)
                
                
    return df1


# In[14]:


def scale_lat_lon(df1):
    scalers_pos = {}
    for i in ["lat","lon"]:
        si = RobustScaler(quantile_range=(10,90))
        values = si.fit_transform(df1[i].values.reshape(-1, 1)).reshape(-1)
        df1[str('scaled_'+i)] = pd.Series(index=df1[i].index,data=values)
        #df1.pop(i)
        scalers_pos[i] = si
    return scalers_pos

def scale_fine_amounts(X_train):
    scalers = {}
    for i in ['fine_amount', 'discount_amount', 'judgment_amount']:
        si = RobustScaler(quantile_range=(0,85),with_centering=False)
        values = si.fit_transform(X_train[i].values.reshape(-1, 1)).reshape(-1)
        X_train[str('scaled_'+i)] = pd.Series(index=X_train[i].index,data=values)
        scalers[i] = si  
    return scalers

# In[15]:


def plot_cross_validation_results(parameters,grid_search,saveplot=""):
    n = len(parameters)
    fig, _ = plt.subplots(n-1,n-1,figsize=(1.8*n,1.8*n),sharex='col',sharey='row')    
    hyper_param = [i.split("__")[-1] for i in parameters.keys()]    
    cv_res = pd.DataFrame(grid_search.cv_results_)
    name_formater = lambda i: 'param_'+ list(parameters.keys())[i]
    
    for i in range(n):
        for j in range(n):
            if j>=i: continue
            cv_pvt = pd.pivot_table(cv_res, values='mean_test_score',index= name_formater(i), columns=name_formater(j))
            z = (i-1)*(n-1) + (j+1)
            ax = plt.subplot(n-1,n-1,z)
            sns.heatmap(cv_pvt,annot=True,cbar=False,vmin=0,vmax=1.0,cmap="cividis",linewidths=.005)
            if i<n-1: ax.get_xaxis().set_visible(False)
            else: ax.set_xlabel(hyper_param[j])
            if j>0: ax.get_yaxis().set_visible(False)
            else: ax.set_ylabel(hyper_param[i])
    plt.subplots_adjust(left=.06,right=.98,top=0.98,bottom=0.07,wspace=0.02,hspace=0.02)
    plt.tight_layout()  
    if saveplot !="":
        plt.savefig("./cv_{}.png".format(saveplot),format='png')
    return

def train_and_cross_validate(model,X_train,y_train,parameters,scoring="accuracy"):
    start_time = time.time()    
    idx_train = X_train.index
    grid= GridSearchCV(model,param_grid=parameters,scoring=scoring,cv=3,n_jobs=-1)
    grid.fit(X_train,y_train.loc[ idx_train])
    print("--- %s seconds ---" % (time.time() - start_time))
    return grid

def confusion_mat(y_true,y_pred):
    cm = pd.DataFrame(data=confusion_matrix(y_true,y_pred))
    cm.set_index(pd.Index(['True 0','True 1']),inplace=True)
    cm.columns=(['Predicted 0','Predicted 1'])
    return cm