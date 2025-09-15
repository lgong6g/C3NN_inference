# hyperparameters are defined in this class


class hyperparameters():
    def __init__(self, learning_rate, flow_hidden_features, num_blocks, flow_num_transforms, flow_training_batch_size,
                 in_channels, num_filts, filter_size, order, save=True, device='cuda'):

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


        #whether to save model
        self.save=save

        #device to use
        self.device=device

    # Name of the model and hyperparameters
    def name_model(self):
        return "order_{:d}_filters_{:d}_size_{:d}_lr_{:.2e}_features_{:d}_blocks_{:d}_transforms_{:d}_batchsize_{:d}".format(self.order,self.num_filts,self.filter_size,self.learning_rate, self.flow_hidden_features, self.num_blocks, self.flow_num_transforms, self.flow_training_batch_size)

# ---------------- Hyperparameters ---------------- #

class hyperparameters_cnn:
    """
    Hyperparameters for a CNN that outputs an embedding vector (bottleneck).

    Required:
      - learning_rate: float
      - in_channels: int (e.g., 1, 3, or 4)
      - num_blocks: int (how many conv blocks; each block downsamples by 2)
      - num_filts: int (base filters; doubled each block)
      - filter_size: int (kernel size, usually odd like 3/5/7/33)
      - bottleneck: int (length of the final embedding vector)

    Optional:
      - use_batchnorm: bool
      - activation: str ("relu" or "gelu")
      - dropout: float (applied in the MLP head)
      - fc_hidden: int or list[int] (hidden size(s) before bottleneck)
      - l2_normalize: bool (if True, return unit-norm embeddings)
      - save: bool
      - device: str
    """
    def __init__(
        self,
        learning_rate,
        in_channels,
        num_blocks,
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

        # CNN hparams (preserving your naming style)
        self.in_channels   = int(in_channels)
        self.num_blocks    = int(num_blocks)
        self.num_filts     = int(num_filts)
        self.filter_size   = int(filter_size)
        self.bottleneck    = int(bottleneck)

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
        return (
            "cnn_blocks_{:d}_basefilts_{:d}_ks_{:d}_bneck_{:d}_lr_{:.2e}_"
            "act_{:s}_bn_{:d}_drop_{:.2f}_fc{:s}_l2_{:d}"
        ).format(
            self.num_blocks,
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

hparams = hyperparameters(learning_rate = 0.00014201803439538936,#0.00017544532449604214,    # Learning rate
                          flow_hidden_features = 64,#128,
                          num_blocks = 4,#5,
                          flow_num_transforms = 4,#3,
                          flow_training_batch_size = 16,#8,
                          in_channels = 4, # not optimzed
                          num_filts = 40, # The number of filters to learn in our model
                          filter_size = 33, # The spatial length of the filters
                          order = 3, # The highest order of correlation measured, not optimized
                          save = True,  
                          device = 'cuda'
                          )


hparams_cnn = hyperparameters_cnn(
     learning_rate=1.4201803439538936e-4,
     in_channels=4,
     num_blocks=2,#4,
     num_filts=10,
     filter_size=33,
     bottleneck=32,
     use_batchnorm=True,
     activation="relu",
     dropout=0.2,
     fc_hidden=[64],
     l2_normalize=False,
     save=True,
     device="cuda",
 )
