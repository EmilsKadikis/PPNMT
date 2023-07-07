from transformers import MarianMTModel
from perturbable_generation_mixin import PerturbableGenerationMixin

class PerturbableMarianMTModel(MarianMTModel, PerturbableGenerationMixin):
    pass