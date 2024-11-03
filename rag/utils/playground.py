import os
from pymongo import MongoClient
import json
import tiktoken


def count_tokens_in_file(file_path: str = 'all_laws.txt', encoding_name: str = "gpt-4o"):
    # Load the tokenizer
    encoding = tiktoken.encoding_for_model(encoding_name)

    # Read the content of the file
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    # Tokenize the text
    tokens = encoding.encode(text)

    # Return the number of tokens
    return len(tokens)




def get_mongo_client():
    # Connect to MongoDB running on localhost
    client = MongoClient("mongodb://localhost:27017/")
    return client

def fetch_all_laws(db_name='law_database_mvp', collection_name='MVP'):
    client = get_mongo_client()
    db = client[db_name]
    collection = db[collection_name]
    # Fetch all documents from the collection
    all_laws = list(collection.find())
    return all_laws

def save_to_text_file(data, file_name='all_laws.txt'):
    with open(file_name, 'w', encoding='utf-8') as f:
        for law in data:
            # Remove the MongoDB specific '_id' field if present
            if '_id' in law:
                del law['_id']
            # Convert the law document to a pretty-formatted JSON string
            law_str = json.dumps(law, ensure_ascii=False, indent=4)
            f.write(law_str)
            f.write('\n\n')  # Add spacing between laws
    print(f"All laws have been saved to {file_name}.")
def tokenize(file_name='all_laws.txt'):
    data = None
    with open(file_name, 'r', encoding='utf-8') as f:
        data = f.read()

def main():
    all_laws = fetch_all_laws()
    save_to_text_file(all_laws)

if __name__ == '__main__':
    # Usage example
    file_path = "../all_laws.txt"
    token_count = count_tokens_in_file(file_path)
    print(f"Number of tokens: {token_count}")
