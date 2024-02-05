import torch
import torch.neuron
from transformers import AutoTokenizer

# Setup some example inputs
sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc")

max_length = 128
paraphrase = tokenizer.encode_plus(
    sequence_0,
    sequence_2,
    max_length=max_length,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)
not_paraphrase = tokenizer.encode_plus(
    sequence_0,
    sequence_1,
    max_length=max_length,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

# Load TorchScript back
neuron_model = torch.jit.load("bert_neuron.pt")

# Convert example inputs to a format that is compatible with TorchScript tracing
example_inputs_paraphrase = (
    paraphrase["input_ids"],
    paraphrase["attention_mask"],
    paraphrase["token_type_ids"],
)
example_inputs_not_paraphrase = (
    not_paraphrase["input_ids"],
    not_paraphrase["attention_mask"],
    not_paraphrase["token_type_ids"],
)

# Warmup
y = neuron_model(*example_inputs_paraphrase)

# Verify the TorchScript works on both example inputs
paraphrase_classification_logits_neuron = neuron_model(*example_inputs_paraphrase)
not_paraphrase_classification_logits_neuron = neuron_model(*example_inputs_not_paraphrase)
print(paraphrase_classification_logits_neuron)
print(not_paraphrase_classification_logits_neuron)

classes = ["not paraphrase", "paraphrase"]
paraphrase_prediction = paraphrase_classification_logits_neuron[0][0].argmax().item()
not_paraphrase_prediction = not_paraphrase_classification_logits_neuron[0][0].argmax().item()
print(
    'Neuron BERT says that "{}" and "{}" are {}'.format(
        sequence_0, sequence_2, classes[paraphrase_prediction]
    )
)
print(
    'Neuron BERT says that "{}" and "{}" are {}'.format(
        sequence_0, sequence_1, classes[not_paraphrase_prediction]
    )
)
