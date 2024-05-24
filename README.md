# neuroflow

Ever wanted to implement the [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear) layer in PyTorch from scratch? No? Great, now you can!

This project started as a port of [karpathy/micrograd](https://github.com/karpathy/micrograd) to JavaScript for educational purposes. The API has been tweaked to provide a useful foundation for reimplementing arbitrary parts of other libraries like PyTorch and TensorFlow, and it includes a host module to execute the code as part of a neural network.

The library prioritizes readability and understanding, and does not take performance considerations into account. As such, it is not suitable for production applications. For production use cases, consider using libraries like [tensorflow-js](https://www.tensorflow.org/js/guide) instead.

## Installation

You can install neuroflow using npm:

```bash
npm install @andypai/neuroflow
```

## Basic Usage

Here's an example of how you can train a simple neural network using the `Layer` and `Sequential` classes

```js
import { Sequential, Layer } from 'neuroflow'

// Specify the architecture
const layer1 = new Layer({ numOfInputs: 3, numOfNeurons: 4 })
const layer2 = new Layer({ numOfInputs: 4, numOfNeurons: 3 })
const layer3 = new Layer({ numOfInputs: 3, numOfNeurons: 1, activation: 'linear' })
const model = new Sequential({
  layers: [layer1, layer2, layer3]
})

const xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]

const ys = [1.0, -1.0, -1.0, 1.0]

const range = (n) => [...Array(n).keys()]

range(200).forEach(() => {
  // forward pass
  const yPredictions = xs.map((x) => model.forward(x))

  // mean squared error (mse)
  const loss = ys.reduce(
    (acc, yTarget, i) => yPredictions[i].sub(yTarget).pow(2).add(acc),
    0,
  )

  // backward pass
  model.zeroGrad()
  loss.backward()

  // update weights
  const learingRate = 0.01
  model.parameters().forEach((p) => {
    p.data -= learingRate * p.grad
  })
})

model.forward([2.0, 3.0, -1.0]) // Output: 1
```

## License

This project is licensed under the MIT License.
