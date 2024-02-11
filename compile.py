import torch
import numpy as np
import os
import torch_neuron
from torchvision import models
from model import GPTConfig, GPT
import tokenizer, transformers



## Load a pretrained nanoGPT model
model = models.resnet50(pretrained=True)

seed_text = "Once upon a time, there was a"
input_ids = tokenizer.encode(seed_text, return_tensors="pt")


## Tell the model we are using it for evaluation (not training)
model.eval()
model_neuron = torch.neuron.trace(model, example_inputs=input_ids)

## Export to saved model
model_neuron.save("resnet50_neuron.pt")
