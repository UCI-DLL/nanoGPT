
import time

import numpy as np
import torch
import torch.neuron
from joblib import Parallel, delayed
from tqdm import tqdm
from transformers import AutoTokenizer

# Sample inputs
positive_example = "This is a really great restaurant, I loved it"
negative_example = "I've never eaten so bad in my life"

# Build tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Convert sample inputs to a format that is compatible with TorchScript tracing
positive = tokenizer(positive_example, return_tensors="pt")
negative = tokenizer(negative_example, return_tensors="pt")
positive_input = (
    positive["input_ids"],
    positive["attention_mask"],
    positive["token_type_ids"],
)
negative_input = (
    negative["input_ids"],
    negative["attention_mask"],
    negative["token_type_ids"],
)

# Load model
neuron_model = torch.jit.load("bert_yelp_neuron.pt")

# Predict samples
positive_logits = neuron_model(*positive_input)
negative_logits = neuron_model(*negative_input)

positive_logits = positive_logits["logits"].cpu().detach().numpy()
negative_logits = negative_logits["logits"].cpu().detach().numpy()

print(positive_logits)
print(negative_logits)

classes = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
positive_prediction = positive_logits[0].argmax()
negative_prediction = negative_logits[0].argmax()
print('Neuron BERT says that "{}" is {}'.format(positive_example, classes[positive_prediction]))
print('Neuron BERT says that "{}" is {}'.format(negative_example, classes[negative_prediction]))


def inference_latency(model, *inputs):
    start = time.time()
    _ = model(*inputs)
    return time.time() - start

num_iterations = 100000
num_threads = 16

t = tqdm(range(num_iterations), position=0, leave=True)
latency = Parallel(n_jobs=num_threads, prefer="threads")(
    delayed(inference_latency)(neuron_model, *positive_input) for i in t
)

p50 = np.quantile(latency, 0.50) * 1000
p95 = np.quantile(latency, 0.95) * 1000
p99 = np.quantile(latency, 0.99) * 1000
avg_throughput = t.total / t.format_dict["elapsed"]
print(f"Avg Throughput: :{avg_throughput:.1f}")
print(f"50th Percentile Latency:{p50:.1f} ms")
print(f"95th Percentile Latency:{p95:.1f} ms")
print(f"99th Percentile Latency:{p99:.1f} ms")
