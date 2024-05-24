# neuralflow-js

Ever wanted to implement the [Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear) layer from scratch? No? Well now you can!

This project started as port of [karpathy/micrograd](https://github.com/karpathy/micrograd) to JavaScript for educational purposes. I've since tweaked the API to make it a useful foundation for reimplement for arbitrary parts of other libraries like PyTorch and TensorFlow and having a host module to execute it as part of a neural network.

The library optimizes for readability and understanding and takes no performance considerations into account. As such, it is not useful for production applications but [tensorflow-js](https://www.tensorflow.org/js/guide) and others are already great for purpose.

## Basic Usage

Here's how you can train a simple neural network using the `Layer` and `Sequential` classes

```js
import Sequential from './sequential.js'
import Layer from './layer.js'

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

model.forward([2.0, 3.0, -1.0]) // 1
```

## License

MIT