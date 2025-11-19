# hyperparameters are defined in this class


class hyperparameters():
    def __init__(self, learning_rate, flow_hidden_features, num_blocks, flow_num_transforms, flow_training_batch_size,
                 in_channels, num_filts, filter_size, order, out_dim, save=True, device='cuda', model='w_std'):

        # Learning rate
        self.learning_rate = learning_rate
        # flow hparams        
        self.flow_hidden_features = flow_hidden_features
        self.num_blocks = num_blocks
        self.flow_num_transforms = flow_num_transforms
        self.flow_training_batch_size = flow_training_batch_size
        # C3NN hparams
        self.in_channels = in_channels
        self.num_filts = num_filts
        self.filter_size = filter_size
        self.order = order
        self.out_dim = out_dim


        #whether to save model
        self.save=save

        #device to use
        self.device=device

        # which C3NN version
        self.model=model

    # Name of the model and hyperparameters
    def name_model(self):
        return "order_{:d}_filters_{:d}_size_{:d}_lr_{:.2e}_features_{:d}_blocks_{:d}_transforms_{:d}_batchsize_{:d}".format(self.order,self.num_filts,self.filter_size,self.learning_rate, self.flow_hidden_features, self.num_blocks, self.flow_num_transforms, self.flow_training_batch_size)


# ---------------- Hyperparameters ---------------- #
class hyperparameters_cnn:
    """
    Hyperparameters for a CNN that outputs an embedding vector (bottleneck),
    with two block types: pooled and non-pooled.
    Required:
      - learning_rate: float
      - in_channels: int
      - num_blocks: int               # number of conv blocks WITH MaxPool
      - num_blocks_nopool: int        # number of conv blocks WITHOUT MaxPool
      - num_filts: int                # base filters; doubled each block across total depth
      - filter_size: int              # kernel size (e.g., 3/5/7/33)
      - bottleneck: int               # length of the final embedding vector
    Optional:
      - use_batchnorm: bool
      - activation: str ("relu" or "gelu")
      - dropout: float                # applied in the MLP head
      - fc_hidden: int or list[int]   # hidden size(s) before bottleneck
      - l2_normalize: bool
      - save: bool
      - device: str
    """
    
    def __init__(
        self,
        learning_rate,
        in_channels,
        num_blocks,
        num_blocks_nopool,      # NEW
        num_filts,
        filter_size,
        bottleneck,
        use_batchnorm=True,
        activation="relu",
        dropout=0.0,
        fc_hidden=128,
        l2_normalize=False,
        save=True,
        device="cuda",
    ):
        # Training hparams
        self.learning_rate = float(learning_rate)
        # CNN hparams
        self.in_channels        = int(in_channels)
        self.num_blocks         = int(num_blocks)            # with MaxPool
        self.num_blocks_nopool  = int(num_blocks_nopool)     # without MaxPool (NEW)
        self.num_filts          = int(num_filts)
        self.filter_size        = int(filter_size)
        self.bottleneck         = int(bottleneck)
        # Head / regularization
        self.use_batchnorm = bool(use_batchnorm)
        self.activation    = str(activation)
        self.dropout       = float(dropout)
        self.fc_hidden     = fc_hidden  # int or list[int]
        self.l2_normalize  = bool(l2_normalize)
        # bookkeeping
        self.save   = bool(save)
        self.device = str(device)
    
    def name_model(self):
        total_blocks = self.num_blocks + self.num_blocks_nopool
        return (
            "cnn_blocksP_{:d}_blocksNP_{:d}_total_{:d}_basefilts_{:d}_ks_{:d}_"
            "bneck_{:d}_lr_{:.2e}_act_{:s}_bn_{:d}_drop_{:.2f}_fc{:s}_l2_{:d}"
        ).format(
            self.num_blocks,
            self.num_blocks_nopool,
            total_blocks,
            self.num_filts,
            self.filter_size,
            self.bottleneck,
            self.learning_rate,
            self.activation.lower(),
            int(self.use_batchnorm),
            self.dropout,
            (str(self.fc_hidden) if isinstance(self.fc_hidden, int) else str(list(self.fc_hidden))),
            int(self.l2_normalize),
        )


#--- HYPERPARAMETER CHOICES ---#

hparams = hyperparameters(learning_rate = 1e-3,  #1.4201803439538936e-4, # Learning rate
                          flow_hidden_features = 128,
                          num_blocks = 4,
                          flow_num_transforms = 15,
                          flow_training_batch_size = 32,
                          in_channels = 4, # not optimzed
                          num_filts = 25, # The number of filters to learn in our model
                          filter_size = 37, # The spatial length of the filters
                          order = 2, # The highest order of correlation measured, not optimized
                          out_dim = 150, # The output dimension from the final FC layer
                          save = True,  
                          device = 'cuda',
                          model = 'w_std'  # default, w_std or hybrid
                          )


hparams_cnn = hyperparameters_cnn(learning_rate = 1e-4,
                                  in_channels = 4,
                                  num_blocks = 4,
                                  num_blocks_nopool = 2,
                                  num_filts = 32,
                                  filter_size = 7,
                                  bottleneck = 32,
                                  use_batchnorm = True,
                                  activation = "relu",
                                  #dropout=0.2,
                                  fc_hidden = [128, 128],
                                  l2_normalize = False,
                                  save = False,
                                  device = "cuda",
                                  )
