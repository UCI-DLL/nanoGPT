import torch
import numpy as np
import os
import torch_neuron
from torchvision import models
from model import GPTConfig, GPT



## Load a pretrained nanoGPT model
model = GPT.from_pretrained(init_from, dict(dropout=0.0))

seed_text = "Once upon a time, there was a"
input_ids = tokenizer.encode(seed_text, return_tensors="pt")


## Tell the model we are using it for evaluation (not training)
model.eval()
model_neuron = torch.neuron.trace(model, example_inputs=input_ids)

## Export to saved model
model_neuron.save("resnet50_neuron.pt")
