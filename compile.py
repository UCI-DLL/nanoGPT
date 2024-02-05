
import torch
import torch.neuron
from transformers import AutoConfig, AutoTokenizer, BertForSequenceClassification

model_id = "bert-base-cased"

# Build tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)

config = AutoConfig.from_pretrained(model_id)
config.num_labels = 5
model = BertForSequenceClassification(config=config)
checkpoint = torch.load("./model.pt")
model.load_state_dict(checkpoint["state_dict"])

# Sample inputs
positive_example = "This is a really great restaurant, I loved it"
negative_example = "I've never eaten so bad in my life"

positive = tokenizer(positive_example, return_tensors="pt")
negative = tokenizer(negative_example, return_tensors="pt")

# Convert sample inputs to a format that is compatible with TorchScript tracing:
# Only Tensors and (possibly nested) Lists, Dicts, and Tuples of Tensors can be traced
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

# Predict sample inputs
positive_logits = model(*positive_input)
negative_logits = model(*negative_input)
print(positive_logits)
print(negative_logits)

classes = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
positive_prediction = positive_logits[0][0].argmax()
negative_prediction = negative_logits[0][0].argmax()
print('Original model says that "{}" is {}'.format(positive_example, classes[positive_prediction]))
print('Original model says that "{}" is {}'.format(negative_example, classes[negative_prediction]))

# Convert model with Neuron
num_neuron_cores = 16  # for inf1.6xlarge
neuron_model = torch.neuron.trace(
    model,
    positive_input,
    strict=False,
    compiler_args=["--neuroncore-pipeline-cores", str(num_neuron_cores)],
)
neuron_model.save("bert_yelp_neuron.pt")
