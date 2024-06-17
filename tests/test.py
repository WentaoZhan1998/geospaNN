import torch
import geospaNN
import numpy as np
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img
import geopandas as gpd
from shapely.geometry import Point
from scipy import spatial, interpolate
import warnings
warnings.filterwarnings("ignore")

url = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_nation_20m.zip"
us = gpd.read_file(url).explode()
us = us.loc[us.geometry.apply(lambda x: x.exterior.bounds[2])<-60]

df_covariates = pd.read_csv('./data/covariate0605.csv')
df_pm25 = pd.read_csv('./data/pm25_0605.csv')
df_pm25 = df_pm25.loc[df_pm25.Latitude < 50]

x_min,y_min,x_max,y_max = np.array([np.min(df_covariates['long']), np.min(df_covariates['lat']),
    np.max(df_covariates['long']), np.max(df_covariates['lat'])])
arr1 = np.mgrid[x_min:x_max:101j, y_min:y_max:101j]

# extract the x and y coordinates as flat arrays
arr1x = np.ravel(arr1[0])
arr1y = np.ravel(arr1[1])
# using the X and Y columns, build a dataframe, then the geodataframe
df = pd.DataFrame({'X':arr1x, 'Y':arr1y})
df['coords'] = list(zip(df['X'], df['Y']))
df['coords'] = df['coords'].apply(Point)

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.X, y=df.Y),crs = us.crs)
inUS = gdf['geometry'].apply(lambda s: s.within(us.geometry.unary_union))

lonlat_pm25=df_pm25.values[:,[1,2]]
near = df_covariates.values[:,[1,2]]
tree = spatial.KDTree(list(zip(near[:,0].ravel(), near[:,1].ravel())))
idx = tree.query(lonlat_pm25)[1]
df_pm25_mean = df_pm25.assign(neighbor = idx).groupby('neighbor')['PM25'].mean()
idx_new = df_pm25_mean.index.values
pm25 = df_pm25_mean.values
z = pm25[:,None]

lon = df_covariates.values[:,1]
lat = df_covariates.values[:,2]

f = interpolate.Rbf(lon[idx_new], lat[idx_new], z, function = 'inverse')
x_test = gdf.loc[inUS,:].X
y_test = gdf.loc[inUS,:].Y
z_test = f(x_test, y_test)

lon = df_covariates.values[:,1]
lat = df_covariates.values[:,2]
covariates = df_covariates.values[:,3:]
normalized_lon = (lon-min(lon))/(max(lon)-min(lon))
normalized_lat = (lat-min(lat))/(max(lat)-min(lat))
normalized_x_test = (x_test-min(lon))/(max(lon)-min(lon))
normalized_y_test = (y_test-min(lat))/(max(lat)-min(lat))

s_obs = np.vstack((normalized_lon[idx_new],normalized_lat[idx_new])).T
X = covariates[idx_new,:]
normalized_X = X
for i in range(X.shape[1]):
    normalized_X[:,i] = (X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i]))

X = normalized_X
Y = z.reshape(-1)
coord = s_obs

data_PM25 = pd.read_csv("./data/Normalized_PM2.5_20190605.csv")
data_PM25

X = torch.from_numpy(data_PM25[['precipitation', 'temperature', 'air pressure', 'relative humidity', 'U-wind', 'V-wind']].to_numpy()).float()
Y = torch.from_numpy(data_PM25[['PM 2.5']].to_numpy().reshape(-1)).float()
coord = torch.from_numpy(data_PM25[['longitude', 'latitude']].to_numpy()).float()

p = X.shape[1]

n = X.shape[0]
nn = 20
batch_size = 50

data = geospaNN.make_graph(X, Y, coord, nn)

torch.manual_seed(2024)
np.random.seed(0)
data_train, data_val, data_test = geospaNN.split_data(X, Y, coord, neighbor_size = 20,
                                                   test_proportion = 0.5)

start_time = time.time()
mlp_nn = torch.nn.Sequential(
    torch.nn.Linear(p, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 1),
)
nn_model = geospaNN.nn_train(mlp_nn, lr =  0.01, min_delta = 0.001)
training_log = nn_model.train(data_train, data_val, data_test)
theta0 = geospaNN.theta_update(torch.tensor([1, 1.5, 0.01]), mlp_nn(data_train.x).squeeze() - data_train.y, data_train.pos, neighbor_size = 20)
mlp_nngls = torch.nn.Sequential(
    torch.nn.Linear(p, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 20),
    torch.nn.ReLU(),
    torch.nn.Linear(20, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1),
)
model = geospaNN.nngls(p=p, neighbor_size=nn, coord_dimensions=2, mlp=mlp_nngls, theta=torch.tensor(theta0))
nngls_model = geospaNN.nngls_train(model, lr =  0.01, min_delta = 0.001)
np.random.seed(2024)
training_log = nngls_model.train(data_train, data_val, data_test,
                                 Update_init = 20, Update_step = 10, seed = 2024)
end_time = time.time()

if batch_size is None:
    batch_size = int(data_train.x.shape[0] / 10)
torch.manual_seed(2024)
train_loader = geospaNN.split_loader(data_train, batch_size)
training_log = {'val_loss': [], 'est_loss': [], 'sigma': [], 'phi': [], 'tau': []}
for epoch in range(100):
    # Train for one epoch
    w = data_train.y - nngls_model.model.estimate(data_train.x)
    nngls_model.model.train()
    nngls_model.model.theta.requires_grad = False
    if (epoch >= 20) & (epoch % 10 == 0):
        nngls_model.theta_update(w, data_train)

    for batch_idx, batch in enumerate(train_loader):
        nngls_model.optimizer.zero_grad()
        decorrelated_preds, decorrelated_targets, est = nngls_model.model(batch)
        loss = torch.nn.functional.mse_loss(decorrelated_preds[:batch_size], decorrelated_targets[:batch_size])
        loss.backward()
        nngls_model.optimizer.step()
    # Compute estimations on held-out test set
    nngls_model.model.eval()
    _, _, val_est = nngls_model.model(data_val)
    val_loss = torch.nn.functional.mse_loss(val_est, data_val.y).item()
    nngls_model.lr_scheduler(val_loss)
    nngls_model.early_stopping(val_loss)
    if nngls_model.early_stopping.early_stop:
        print('End at epoch' + str(epoch))
        break
    training_log["val_loss"].append(val_loss)
    training_log["sigma"].append(nngls_model.model.theta[0].item())
    training_log["phi"].append(nngls_model.model.theta[1].item())
    training_log["tau"].append(nngls_model.model.theta[2].item())
    if data_test is None:
        _, _, test_est = nngls_model.model(data_test)
        est_loss = torch.nn.functional.mse_loss(test_est, data_test.y).item()
        training_log["est_loss"].append(est_loss)