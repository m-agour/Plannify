import torch
from model.cnn_models import CentroidRegressor, XXYYRegressor

# loading models

# loading centroid model
regressor_model = CentroidRegressor()
regressor_model.load_state_dict(
    torch.load('model/models/centroid_regressor_state_dict.pth')
)
regressor_model.eval()

# loading boundaries model
boundaries_model = XXYYRegressor()
boundaries_model.load_state_dict(
    torch.load("model/models/xxyy_model_state_dict.pth")
)
boundaries_model.eval()
