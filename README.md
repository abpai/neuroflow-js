# NeuroFlow

[![NPM Version](https://img.shields.io/npm/v/@andypai/neuroflow.svg)](https://www.npmjs.com/package/@andypai/neuroflow)
[![License](https://img.shields.io/npm/l/@andypai/neuroflow.svg)](https://github.com/yourusername/neuroflow/blob/main/LICENSE)

NeuroFlow is a JavaScript library that allows you to implement neural network layers and architectures from scratch, similar to the [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear) layer in PyTorch. It provides a foundation for reimplementing arbitrary parts of other libraries like PyTorch and TensorFlow, and includes a host module to execute the code as part of a neural network.

This project started as a port of [karpathy/micrograd](https://github.com/karpathy/micrograd) to JavaScript for educational purposes. The API has been tweaked to prioritize readability and understanding, making it suitable for learning and experimentation. However, it does not take performance considerations into account and is not recommended for production applications. For production use cases, consider using libraries like [TensorFlow.js](https://www.tensorflow.org/js/guide) instead.

Demos: https://neuroflow.andypai.me/

## Installation

You can install NeuroFlow using npm:

```bash
npm install @andypai/neuroflow
```

## Basic Usage

Here's an example of how you can train a simple neural network using the `Layer` and `Sequential` classes for a regression problem:

```js
import { Sequential, Layer } from 'neuroflow'

// Specify the architecture
const layer1 = new Layer({ numOfInputs: 3, numOfNeurons: 4 })
const layer2 = new Layer({ numOfInputs: 4, numOfNeurons: 3 })
const layer3 = new Layer({
  numOfInputs: 3,
  numOfNeurons: 1,
  activation: 'linear',
})
const model = new Sequential({
  layers: [layer1, layer2, layer3],
})

// Training data
const xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
const ys = [1.0, -1.0, -1.0, 1.0]

// Training loop
const epochs = 15
for (let epoch = 0; epoch < epochs; epoch++) {
  // Forward pass
  const yPredictions = xs.map((x) => model.forward(x))

  // Mean squared error (MSE) loss
  const loss = ys.reduce(
    (acc, yTarget, i) => yPredictions[i].sub(yTarget).pow(2).add(acc),
    0,
  )

  // Backward pass
  model.zeroGrad()
  loss.backward()

  // Update weights
  const learningRate = 0.01
  model.parameters().forEach((p) => {
    p.data -= learningRate * p.grad
  })

  console.info(`Epoch: ${epoch + 1}, Loss: ${loss.data}`)
}

// Inference
model.forward([2.0, 3.0, -1.0]) // Output: 1
```

The full example is available [here](./src/examples/linear-regression.js)


## Binary Classifier

NeuroFlow can be used to train a binary classifier. Here are the key differences compared to the regression example:

```js

const xs = [
  [1.0, 2.0],
  [2.0, 3.0],
  [3.0, 1.0],
  [4.0, 2.0],
]

const ys = [1.0, 1, -1.0, -1.0] // y > x => 1

let learningRate = 0.5
const alpha = 1e-4
const epochs = 15

range(epochs).forEach((epoch) => {
  const yPredictions = xs.map((input) => model.forward(input))

  const losses = yPredictions.reduce((acc, yPrediction, index) => {
    // hinge loss: max(0, 1 - y * ŷ)
    const loss = yPrediction.mul(-ys[index]).add(1).relu()
    return loss.add(acc)
  }, new Value(0))

  const dataLoss = losses.div(yPredictions.length)

  // regularization loss
  const regLoss = new Value(alpha).mul(
    model.parameters().reduce((acc, p) => acc.add(p.pow(2)), new Value(0)),
  )

  const totalLoss = dataLoss.add(regLoss)

  const correct = yPredictions.filter(
    (y, index) => y.data > 0 === ys[index] > 0,
  ).length
  const accuracy = correct / yPredictions.length

  model.zeroGrad()
  totalLoss.backward()

  learningRate = 0.5 - (0.45 * epoch) / epochs

  model.parameters().forEach((p) => {
    p.data -= learningRate * p.grad
  })

  console.info(
    `Epoch: ${epoch + 1}, Loss: ${totalLoss.data}, Accuracy: ${accuracy}`,
  )
})

model.forward([1.0, 2.0]) // Output: 1
```

The full example is available [here](./src/examples/binary-classification.js)

## Multi-class Classification

To handle multi-class classification (3 or more classes) using softmax and cross-entropy loss, here are the key differences:

```js
// last layer should use `softmax` activiation
const layer3 = new Layer({
  numOfInputs: 16,
  numOfNeurons: 3,
  activation: 'softmax',
})

// use cross entropy loss
const crossEntropyLoss = (predictions, labels) => {
  const n = predictions.length
  return predictions
    .reduce((acc, pred, i) => {
      const label = labels[i]
      const loss = pred
        .map((p, j) => new Value(-label[j]).mul(p.log()))
        .reduce((a, b) => a.add(b), new Value(0))
      return acc.add(loss)
    }, new Value(0))
    .div(n)
}
```

The full example is available [here](./src/examples/multiclass-classification.js)

## Roadmap

- [x] Regression
- [x] Binary Classification
- [x] Multi-class Classification
- [ ] Autoencoder
- [ ] Convolutional Neural Network (CNN)
- [ ] Recurrent Neural Network (RNN)
- [ ] Long Short-Term Memory (LSTM)
- [ ] Bidirectional LSTM/GRU
- [ ] Attention Mechanisms
- [ ] Transformer
- [ ] Siamese Networks
- [ ] Variational Autoencoder (VAE)
- [ ] Generative Adversarial Network (GAN)
- [ ] Graph Neural Networks (GNN)
- [ ] Capsule Networks
- [ ] Reinforcement Learning

## License

This project is licensed under the MIT License.
