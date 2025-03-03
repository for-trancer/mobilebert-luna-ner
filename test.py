import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Path to your saved model directory
model_dir = "."

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForTokenClassification.from_pretrained(model_dir)

# Load the label mapping from intent_mapping.json
with open("intent_mapping.json", "r") as f:
    id2label_mapping = json.load(f)

# Sample sentence to test
sentence = "set an alarm for 4:00 pm tommorow"

# Tokenize the input sentence with offset mapping
inputs = tokenizer(
    sentence,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_length=128,
    return_offsets_mapping=True
)

# Extract and remove the offset mapping from inputs
offset_mapping = inputs.pop("offset_mapping")

# Run model inference: get logits for each token
outputs = model(**inputs)
logits = outputs.logits  # shape: (batch_size, seq_len, num_labels)

# Compute probabilities with softmax
probs = F.softmax(logits, dim=2)

# Get the predicted label IDs for each token
predictions = torch.argmax(logits, dim=2)[0].tolist()

# Get the predicted probabilities for each token (list of lists)
predicted_probs = probs[0].tolist()

# Convert input IDs back to tokens
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

results = []
# Loop through tokens along with their predicted label and offset mapping.
for i, (token, pred_id, prob_list, offset) in enumerate(zip(tokens, predictions, predicted_probs, offset_mapping[0].tolist())):
    # Skip special tokens or tokens with no character span (e.g. padding)
    if token in tokenizer.all_special_tokens or offset[0] == offset[1]:
        continue
    # Map the predicted label id to the original label using the JSON mapping.
    label = id2label_mapping.get(str(pred_id), "Unknown")
    # Skip tokens predicted as "O"
    if label == "O":
        continue
    # Get the probability score of the predicted label
    score = prob_list[pred_id]
    # Build the dictionary for this token.
    result = {
        "word": token,
        "score": score,
        "entity": label,
        "index": i,
        "start": offset[0],
        "end": offset[1]
    }
    results.append(result)

# Print the results as a JSON-formatted string
print(json.dumps(results, indent=4))
