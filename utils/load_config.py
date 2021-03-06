import yaml


class Configuration(object):
    def __init__(self, config):
        self.encoder_config = EncoderConfig(config)
        self.decoder_config = DecoderConfig(config)
        self.set_generator_config = SetGeneratorConfig(config)

        self.glob = config["Global"]
        modules = config["Modules"]

        self.latent_dim = self.glob['latent_dim']
        self.cosine_channels = self.glob['cosine_channels']

class EncoderConfig(object):
    def __init__(self, config):
        encoder = config["Encoder"]
        glob = config["Global"]

        self.in_dim = glob['dataset_max_n']
        self.initial_mlp_layers = encoder['initial_mlp_layers']
        self.hidden_initial = encoder['hidden_initial']

        self.layers = encoder['layers']
        self.hidden = encoder['hidden']
        self.use_residual = encoder['use_residual']
        self.use_bn = encoder['use_bn']

        self.aggregator = encoder["aggregator"]

        self.final_mlp_layers = encoder['final_mlp_layers']
        self.hidden_final = encoder['hidden_final']
        self.latent_dim = glob['latent_dim']
        self.cosine_channels = glob['cosine_channels'] if config['SetGenerator']['name'] == 'TopKGenerator' else 0

        self.modules_config = ModuleConfig(config)


class DecoderConfig(object):
    def __init__(self, config):
        decoder = config["Decoder"]
        glob = config["Global"]

        self.max_n = glob['dataset_max_n']
        self.latent_dim = glob['latent_dim']
        self.cosine_channels = glob['cosine_channels'] if config['SetGenerator']['name'] == 'TopKGenerator' else 0
        self.set_channels = glob['set_channels']
        self.hidden_initial = decoder['hidden_initial']
        self.initial_mlp_layers = decoder['initial_mlp_layers']
        self.modulation = decoder['modulation']
        self.layers = decoder['layers']
        self.hidden = decoder['hidden']
        self.use_residual = decoder['use_residual']
        self.use_bn = decoder['use_bn']

        self.final_mlp_layers = decoder['final_mlp_layers']
        self.hidden_final = decoder['hidden_final']

        self.modules_config = ModuleConfig(config)


class SetGeneratorConfig(object):
    def __init__(self, config):
        set_gen = config['SetGenerator']
        glob = config['Global']
        self.name = set_gen['name']
        self.latent_dim = glob['latent_dim']
        self.set_channels = glob['set_channels']
        self.cosine_channels = glob['cosine_channels'] if set_gen['name'] == 'TopKGenerator' else 0
        self.learn_from_latent = set_gen['learn_from_latent']
        self.n_distribution = set_gen['n_distribution']
        self.num_mlp_layers = set_gen['num_mlp_layers']
        self.extrapolation_n = set_gen['extrapolation_n']
        self.mlp_gen_hidden = set_gen['mlp_gen_hidden']
        self.hidden = set_gen['hidden']
        self.dataset_max_n = glob['dataset_max_n']


class ModuleConfig(object):
    def __init__(self, config):
        modules = config['Modules']

        self.preprocessing_steps = modules["Set2Set"]["processing_steps"]
        self.average_n = modules["PNA"]["average_n"]
        self.num_mlp_layers = modules["MLP"]["num_mlp_layers"]
        self.hidden_mlp = modules["MLP"]['hidden_mlp']

        transformer = modules['Transformer']
        self.n_heads = transformer['n_heads']
        self.head_width = transformer['head_width']
        self.dim_feedforward = transformer['dim_feedforward']
        self.residuals = transformer['residuals']
