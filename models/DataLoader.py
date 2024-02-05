from utils.base_dataloader import load_base_data, load_auxiliary_data

from .GraphWaveNet.data_loader import load_gwnet_base_data, load_gwnet_auxiliary_data
from .GMAN.data_loader import load_gman_base_data, load_gman_auxiliary_data
from .STGNCDE.data_loader import load_stgncde_base_data, load_stgncde_auxiliary_data
from .D2STGNN.data_loader import load_d2stgnn_base_data, load_d2stgnn_auxiliary_data
from .STWave.data_loader import load_stwave_base_data, load_stwave_auxiliary_data


def data_loader(model_name, backbone_name):
    if backbone_name == 'astgcn':
        if model_name == 'gwnet':
            loader = load_gwnet_base_data
        elif model_name == 'gman':
            loader = load_gman_base_data
        elif model_name == 'stgncde':
            loader = load_stgncde_base_data
        elif model_name == 'd2stgnn':
            loader = load_d2stgnn_base_data
        elif model_name == 'stwave':
            loader = load_stwave_base_data
        else:
            loader = load_base_data
    elif backbone_name == 'gwnet':
        if model_name == 'gwnet':
            loader = load_gwnet_auxiliary_data
        elif model_name == 'gman':
            loader = load_gman_auxiliary_data
        elif model_name == 'stgncde':
            loader = load_stgncde_auxiliary_data
        elif model_name == 'd2stgnn':
            loader = load_d2stgnn_auxiliary_data
        elif model_name == 'stwave':
            loader = load_stwave_auxiliary_data
        else:
            loader = load_auxiliary_data
    return loader