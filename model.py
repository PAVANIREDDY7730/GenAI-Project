import json

def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        dataset = json.load(file)
    return dataset

def genai_engine(prompt):
    # Load the dataset from the JSON file
    dataset = load_dataset("emojidataset [MConverter.eu].json")
    
    # Check if the prompt matches any input message in the dataset
    for data in dataset:
        if prompt.lower() == data["input"].lower():
            return data['emoji']  # Return only the emoji corresponding to the input message
    
    # If no match is found, return a default response
    return "❓"  # Return a question mark emoji for unknown input
