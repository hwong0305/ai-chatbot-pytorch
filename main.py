import json
import os
import random

import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Architecture of ChatBotModel
class ChatbotModel(nn.Module):

    # def __init__(self, input_size, output_size):
    #     super(ChatbotModel, self).__init__()

    #     self.fc1 = nn.Linear(input_size, 128)  # 128 Neurons. a Keros or dense layer
    #     self.fc2 = nn.Linear(128, 64)  # Second Layer
    #     self.fc3 = nn.Linear(64, output_size)
    #     self.relu = nn.ReLU()  # Activation function to break linearity
    #     self.dropout = nn.Dropout(0.5)  # Regularization

    def __init__(self, vocab_size, embedding_dim, output_size):
        super(ChatbotModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        averaged = torch.mean(embedded, dim=1)
        # x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        # x = self.fc3(x)  # Will apply softmax instead of relu
        x = self.relu(self.fc1(averaged))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x


class ChatbotAssistant:

    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        self.max_seq_length = 20

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_response = {}

        self.function_mappings = function_mappings
        self.word_to_index = {}

        self.X = None  # Usually a matrix
        self.y = None  # Usually a vector

    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmatizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmatizer.lemmatize(word.lower()) for word in words]

        return words

    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]

    def parse_intents(self):
        if os.path.exists(self.intents_path):
            with open(self.intents_path, "r") as f:
                intents_data = json.load(f)

            for intent in intents_data["intents"]:
                if intent["tag"] not in self.intents:
                    self.intents.append(intent["tag"])
                    self.intents_response[intent["tag"]] = intent["responses"]

                for pattern in intent["patterns"]:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append((pattern_words, intent["tag"]))

                self.vocabulary = sorted(set(self.vocabulary))
                self.word_to_index = {
                    word: idx for idx, word in enumerate(self.vocabulary)
                }

    def prepare_data(self):
        bags = (
            []
        )  # Being replaced with sequences to use model that can determine similarity
        sequences = []
        indices = []

        for document in self.documents:
            words = document[0]
            # bag = self.bag_of_words(words, self.vocabulary)
            seq = [self.word_to_index.get(word, 0) for word in words]
            # pad and truncate to size
            seq = seq[: self.max_seq_length] + [0] * (self.max_seq_length - len(seq))
            sequences.append(seq)

            # intent_index = self.intents.index(document[1])

            # bags.append(bag)
            # indices.append(intent_index)
            indices.append(self.intents.index(document[1]))

        self.X = np.array(sequences, dtype=np.long)
        self.y = np.array(indices)

    def train_model(self, batch_size, lr, epochs):
        # X_tensor = torch.tensor(self.X, dtype=torch.float32)
        X_tensor = torch.tensor(self.X, dtype=torch.long)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # self.model = ChatbotModel(self.X.shape[1], len(self.intents))
        self.model = ChatbotModel(
            vocab_size=len(self.vocabulary),
            embedding_dim=100,
            output_size=len(self.intents),
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss

            print(f"Epoch {epoch+1}: Loss: {running_loss/ len(loader):.4f}")

    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, "w") as f:
            # json.dump({"input_size": self.X.shape[1], "output": len(self.intents)}, f)
            json.dump(
                {
                    "vocab_size": len(self.vocabulary),
                    "embedding_dim": self.model.embedding.embedding_dim,
                    "output_size": len(self.intents),
                },
                f,
            )

    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, "r") as f:
            dimensions = json.load(f)

        # self.model = ChatbotModel(dimensions["input_size"], dimensions["output"])
        self.model = ChatbotModel(
            vocab_size=dimensions["vocab_size"],
            embedding_dim=dimensions["embedding_dim"],
            output_size=dimensions["output_size"],
        )
        # self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.load_state_dict(torch.load(model_path))

    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        # bag = self.bag_of_words(words)

        pos_tags = nltk.pos_tag(words)
        quantites = []
        for word, tag in pos_tags:
            if tag == "CD" and word.isdigit():
                quantites.append(int(word))
        quantity = quantites[0] if quantites else 3  # Default to 3

        # bag_tensor = torch.tensor([bag], dtype=torch.float32)
        seq = [self.word_to_index.get(word, 0) for word in words]
        seq = seq[: self.max_seq_length] + [0] * (self.max_seq_length - len(seq))
        seq_tensor = torch.tensor([seq], dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            # predictions = self.model(bag_tensor)
            predictions = self.model(seq_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        predicted_intent = self.intents[predicted_class_index]

        if self.function_mappings:
            if predicted_intent in self.function_mappings:
                self.function_mappings[predicted_intent](quantity)

        if self.intents_response[predicted_intent]:
            return random.choice(self.intents_response[predicted_intent])
        else:
            return None


def get_stocks(quantity=3):
    stocks = ["APPL", "META", "NVDA", "GS", "MSFT"]

    print(random.sample(stocks, quantity))


if __name__ == "__main__":

    assistant = ChatbotAssistant(
        "intents.json", function_mappings={"stocks", get_stocks}
    )
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=400)

    assistant.save_model("chatbot_model.pth", "dimensions.json")

    assistant = ChatbotAssistant(
        "intents.json", function_mappings={"stocks": get_stocks}
    )
    assistant.parse_intents()
    assistant.load_model("chatbot_model.pth", "dimensions.json")

    while True:
        message = input("Enter your message:")

        if message == "/quit":
            break

        print(assistant.process_message(message))
