from .free_regression import Regression
from .graphics import plot_expected, plot_residual, plot_prediction_bands
from .models_regression import generate_regression, generate_mlp, generate_mlp_classifier, generate_mlp_semi_classifier, generate_mlp_sigmoid_sum, generate_mlp_normals, generate_distribuction, asymmetric_normal
from .data import MedidasDeMassa, ProdutividadeTrabalhoRemoto, to_dummy, transpose, normalize, data_series, density
from .make_animation import make_animation
