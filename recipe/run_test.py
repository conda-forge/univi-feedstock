# BEGIN UNIVI RELEASE API CHECK
from copy import deepcopy
import tempfile
import torch
import univi
from univi import (ClassHeadConfig, ModalityConfig, UniVIConfig, UniVIMultiModalVAE,
                   UniVIRefiner, RefinementConfig, RNAPreprocessor, ADTPreprocessor,
                   ATACPreprocessor, save_reference, load_reference,
                   predict_feature_perturbation, make_loader, stack_embeddings,
                   load_scnmt_gastrulation_genebody_triplet)

assert univi.__version__ == "1.1.0"
torch.set_num_threads(1)
model = UniVIMultiModalVAE(UniVIConfig(
    latent_dim=3, encoder_batchnorm=False, decoder_batchnorm=True,
    modalities=[ModalityConfig("rna", 4, [8], [8]), ModalityConfig("adt", 3, [8], [8])]))
model.freeze_decoders().train()
assert not model.decoders["rna"].training
before = deepcopy(model.decoders.state_dict())
model.add_classification_head(ClassHeadConfig(
    "celltype", 2, hidden_dims=[4], batchnorm=False, layernorm=True),
    label_names=["A", "B"])
assert all(torch.equal(value, model.decoders.state_dict()[name]) for name, value in before.items())
with tempfile.TemporaryDirectory() as directory:
    save_reference(directory, model, metadata={"check": "conda-forge"})
    restored, _, metadata = load_reference(directory)
    assert restored.head_label_names["celltype"] == ["A", "B"]
    assert metadata["check"] == "conda-forge"
print("UniVI public API and reference bundle checks passed.")
# END UNIVI RELEASE API CHECK
