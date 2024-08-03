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

## Examples

- [x] [Regression](./src/examples/linear-regression.js)
- [x] [Binary Classification](./src/examples/binary-classification.js)
- [x] [Multi-class Classification](./src/examples/multiclass-classification.js)
- [x] [MNIST Classifier](./src/examples/mnist-classifier.js)
- [x] [MNIST Autoencoder](./src/examples/mnist-autoencoder.js)

## License

This project is licensed under the MIT License.
