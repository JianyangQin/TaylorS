from models.ASTGCN.builder import build_astgcn
from models.GraphWaveNet.builder import build_gwnet
from models.GMAN.builder import build_gman
from models.STFGNN.builder import build_stfgnn
from models.STGODE.builder import build_stgode
from models.STGNCDE.builder import build_stgncde
from models.D2STGNN.builder import build_d2stgnn
from models.STWave.builder import build_stwave


def models_builder(train_phase, checkpoint_filename, **args):
    model_name = args['start_up']['base_model_name']

    if model_name == 'astgcn':
        from models.ASTGCN.ASTGCN_r import ASTGCN
        model = build_astgcn(
                             network=ASTGCN,
                             train_phase=train_phase,
                             checkpoint_filename=checkpoint_filename,
                             **args)
    elif model_name == 'gwnet':
        from models.GraphWaveNet.GraphWaveNet import gwnet
        model = build_gwnet(
                            network=gwnet,
                            train_phase=train_phase,
                            checkpoint_filename=checkpoint_filename,
                            **args)
    elif model_name == 'gman':
        from models.GMAN.GMAN import GMAN
        model = build_gman(
                           network=GMAN,
                           train_phase=train_phase,
                           checkpoint_filename=checkpoint_filename,
                           **args)
    elif model_name == 'stfgnn':
        from models.STFGNN.STFGNN import STFGNN
        model = build_stfgnn(
                             network=STFGNN,
                             train_phase=train_phase,
                             checkpoint_filename=checkpoint_filename,
                             **args)
    elif model_name == 'stgode':
        from models.STGODE.STGODE import STGODE
        model = build_stgode(
                             network=STGODE,
                             train_phase=train_phase,
                             checkpoint_filename=checkpoint_filename,
                             **args)
    elif model_name == 'stgncde':
        from models.STGNCDE.STGNCDE import NeuralGCDE
        model = build_stgncde(
                              network=NeuralGCDE,
                              train_phase=train_phase,
                              checkpoint_filename=checkpoint_filename,
                              **args)
    elif model_name == 'd2stgnn':
        from models.D2STGNN.D2STGNN import D2STGNN
        model = build_d2stgnn(
                              network=D2STGNN,
                              train_phase=train_phase,
                              checkpoint_filename=checkpoint_filename,
                              **args)
    elif model_name == 'stwave':
        from models.STWave.STWave import STWave
        model = build_stwave(
                             network=STWave,
                             train_phase=train_phase,
                             checkpoint_filename=checkpoint_filename,
                             **args)
    else:
        raise ValueError("Model {} does not exist".format(model_name))
    return model

def adjust_builder(train_phase, backbone, checkpoint_filename, **args):
    if backbone == 'astgcn':
        from models.AdjustNet.builder import build_adjustnet_astgcn
        from models.AdjustNet.AdjustNet import AdjustNet_withASTGCN
        model = build_adjustnet_astgcn(
            network=AdjustNet_withASTGCN,
            train_phase=train_phase,
            checkpoint_filename=checkpoint_filename,
            **args
        )
    elif backbone == 'gwnet':
        from models.AdjustNet.builder import build_adjustnet_gwnet
        from models.AdjustNet.AdjustNet import AdjustNet_withGWNET
        model = build_adjustnet_gwnet(
            network=AdjustNet_withGWNET,
            train_phase=train_phase,
            checkpoint_filename=checkpoint_filename,
            **args
        )

    return model