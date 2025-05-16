# Implementation of AI Chatbot with PyTorch

This started off as a simple chatbot implementation of an AI Chatbot with Pytorch.

It was done first following the tutorial [here](https://www.youtube.com/watch?v=a040VmmO-AY).

This does not use an LLM wrapper. It implements its own neural network in PyTorch by classifying the intents of user and respond according. The user can extend the functionality by implementing functions for the AI to use when dealing with certain intent. (Maybe add an Akuma functionality 殺意の波動)

The original code is also referenced in the neuronine folder.

## Extended Functionality from Tutorial

I have extended the chatbot functionality by adding Named Entity Recognition (NER) which lets the users specify how many stocks on their portfolio I want to see.

It also uses AI to increase the accuracy of intent recognition by implementing similarity of certain terms by implementing embeddings.

## Further for the project

This can be easily deployed into a FASTAPI server. That's just using the FastAPI to expose certain endpoints and returning `assistant.process(message)` as the API response. Since it's just python code it can be deployed on the web using any VPC.

Further more, what would be more interesting is buidling an UI for this chatbot to make it into a full stack application. It could share code with one of the many slack clone projects on GitHub to prototype fast or use an AI agent to build the UI.
