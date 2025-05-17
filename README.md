# Implementation of AI Chatbot with PyTorch

This started off as a simple chatbot implementation of an AI Chatbot with Pytorch.

It was done first following the tutorial [here](https://www.youtube.com/watch?v=a040VmmO-AY).

This does not use an LLM wrapper. It implements its own neural network in PyTorch by classifying the
intents of user and respond according. The user can extend the functionality by implementing
functions for the AI to use when dealing with certain intent. (Maybe add an Akuma functionality 殺意
の波動)

The original code is also referenced in the neuronine folder.

## Extended Functionality from Tutorial

I have extended the chatbot functionality by adding Named Entity Recognition (NER) which lets the
users specify how many stocks on their portfolio I want to see.

It also uses AI to increase the accuracy of intent recognition by implementing similarity of certain
terms by implementing embeddings.

I have put in the code for a FastAPI server and a React frontend that can talk to the bot.

## Deployment notes

Create a virtual environment and run `pip install -r requirements.txt`

The following NLTK packages need to be downloaded

- `punkt_tab`
- `wordnet`
- `averaged_perceptron_tagger_eng`

Then the FastAPI server can be deployed.

The react server can use `bun run build` with a static adapter to deploy into a S3 bucket and then
use reverse proxy to lin to the backend.
