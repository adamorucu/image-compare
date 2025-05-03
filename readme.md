# Eval models
A place to evaluate different types of deep learning models. We will use weights and biases to track training.
The project trains the model from zero and evaluates it.
We will later use these models and their evaluations as benchmarks and will start developing our own methods on top of those.

Possibility to evaluate models on:
- MNIST
- CIFAR 10

Models available for test:
- MLP of different size
    - 2 * [64]
    - 4 * [64]
    - 4 * [512]
- CNN (shallow and deep)
- ResNet
- MLP-mixer

## Project Structure
- train.py
- eval.py
- models/
    - mlp.py
    - cnn.py
    - mixer.py
    - resnet.py
- run.py
- data/
- config.py
- utils.py