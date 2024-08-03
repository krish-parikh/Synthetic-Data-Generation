from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# enforce_min_max_values: bool = True/False -> each column stays within the min and max values of the real data
# enforce_rounding: bool = True/False -> round the generated values to the nearest integer
# locales: str = 'en_US' -> generate PII data according to specific regions and languages

# numerical_distributions: dict = None -> distribution to use for each numerical column i.e. 'norm' 'beta', 'truncnorm', 'uniform', 'gamma' or 'gaussian_kde'
# default_distribution: str = 'norm' -> distribution to use for numerical columns that are not specified


# batch_size: int = 500 -> number of samples to train on each step, increase to improve training stability

# discriminator_dim: Tuple[int, int] = (256, 256) -> number of layers and the neurons in each layer of the discriminator
# generator_dim: Tuple[int, int] = (256, 256) -> number of layers and the neurons in each layer of the generator
# embedding_dim: int = 128 -> size of the random noise vector, increase to increase capacity of the model to learn complex patterns

# discrinimnator_decay: float = 1e-6 -> decay rate of the discriminator learning rate
# generator_decay: float = 1e-6 -> decay rate of the generator learning rate

# discriminator_lr: float = 2e-4 -> learning rate of the discriminator
# generator_lr: float = 2e-4 -> learning rate of the generator

# discriminator_steps: int = 1 -> number of discriminator updates per generator update, increase to improve stability

# log_frequency: bool = True/False -> whether to use log frequency of categorical levels in conditional sampling.
# pac: int = 10 -> number of samples to group together when applying the discriminator.


class SDG:
    def __init__(self, data):
        self.real = data
        self.fake = None
        self.metadata = None
        self.synthesizer = None

    def metadata(self):
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(self.real)
        return self.metadata
    
    def transform(self):
        scaler = MinMaxScaler()
        self.real = pd.DataFrame(scaler.fit_transform(self.real), columns=self.real.columns)
        return self.real
    
    def GCS(self):
        self.synthesizer = GaussianCopulaSynthesizer(metadata=self.metadata,
                                                     enforce_min_max_values=True,
                                                     enforce_rounding=False,
                                                     numerical_distributions={},
                                                     default_distribution='norm')
        self.synthesizer.fit(self.real)
        self.fake = self.synthesizer.sample(len(self.real))
        print("Evaluation of the model: ", evaluate_quality(self.fake, self.real))
        return self.fake

    def CTGAN(self):
        self.synthesizer = CTGANSynthesizer(metadata=self.metadata,
                                            enforce_min_max_values=True,
                                            enforce_rounding=False,
                                            epochs=300,
                                            batch_size=500,
                                            embedding_dim=128,
                                            generator_dim=(256, 256),
                                            discriminator_dim=(256, 256),
                                            generator_lr=2e-4,
                                            generator_decay=1e-6,
                                            discriminator_lr=2e-4,
                                            discriminator_decay=1e-6,
                                            discriminator_steps=1,
                                            log_frequency=True,
                                            verbose=True,
                                            pac=10)
        self.synthesizer.fit(self.real)

    def T
    

        self.fake.fit(self.real)
        self.fake = self.fake.sample(len(self.real))
        return self.fake
