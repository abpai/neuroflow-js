import { Value } from '@andypai/neuroflow'

import GameEngine from './js/game-engine.js'
import { generateLinearData } from './js/patterns.js'
import range from './js/range.js'
import buildModel from './js/multiclass-classification-model.js'
import visualizeNetwork from './js/network-visualization.js'

const size = Math.min(+document.querySelector('.demo').offsetWidth, 600)

const percent = (v) =>
  new Intl.NumberFormat('en-US', { style: 'percent' }).format(v)

const oneHotEncode = (label, numClasses) => {
  const encoding = Array(numClasses).fill(0)
  encoding[label] = 1
  return encoding
}

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

const oneHotDecode = (values) => {
  const probs = values.map((v) => v.data)
  return probs.indexOf(Math.max(...probs))
}

class MultiClassClassification extends GameEngine {
  constructor({ canvas, lossCanvas }) {
    super({ canvas })
    this.lossCanvas = lossCanvas
    this.ctxLoss = this.lossCanvas.getContext('2d')
    this.lossCanvas.setAttribute('width', size)
    this.lossCanvas.setAttribute('height', 200)

    this.model = buildModel()
    this.training = false
    this.trainId = null
    this.drawId = null
    this.maxEpochs = 500
    this.trainingData = []
    this.lossHistory = []
  }

  mapX(x) {
    return this.padding + x * this.innerWidth
  }

  mapY(y) {
    return this.height - this.padding - y * this.innerHeight
  }

  plotPattern() {
    const [data, labels] = this.trainingData

    data.forEach((point, index) => {
      const [x, y] = point
      const label = labels[index]
      const color =
        (label === 2 && '#3a7ca5') || // blue
        (label === 1 && '#f6511d') || // orange
        '#006632' // green
      this.drawCircle(this.mapX(x), this.mapY(y), 5, color)
    })
  }

  plotBoundaries() {
    const step = 0.05
    const xRange = range(0, 1, step)
    const yRange = range(0, 1, step)

    yRange.forEach((y) => {
      xRange.forEach((x) => {
        const prediction = this.model.forward([x, y])
        const label = oneHotDecode(prediction)
        const color =
          (label === 2 && 'rgb(58 124 165 / 20%)') || // blue
          (label === 1 && 'rgb(246 81 29 / 20%)') || // orange
          'rgb(0 199 97 / 50%)' // green
        this.drawRectangle(
          this.mapX(x),
          this.mapY(y),
          step * this.innerWidth,
          step * this.innerHeight,
          color,
        )
      })
    })
  }

  plotLossCurve() {
    const { lossCanvas, ctxLoss, lossHistory } = this

    if (lossHistory.length < 2) return

    const { width } = lossCanvas
    const { height } = lossCanvas
    const padding = 25
    const plotWidth = width - 2 * padding
    const plotHeight = height - 2 * padding

    // Clear the previous plot
    ctxLoss.clearRect(0, 0, width, height)

    // Set up the drawing area
    ctxLoss.strokeStyle = '#856a5d'
    ctxLoss.lineWidth = 2
    ctxLoss.beginPath()

    // Find the maximum loss for scaling
    const maxLoss = Math.max(...lossHistory)

    lossHistory.forEach((loss, index) => {
      const x = padding + (index / (this.maxEpochs - 1)) * plotWidth
      const y = height - padding - (loss / maxLoss) * plotHeight

      if (index === 0) {
        ctxLoss.moveTo(x, y)
      } else {
        ctxLoss.lineTo(x, y)
      }
    })

    ctxLoss.stroke()
    ctxLoss.closePath()

    // Draw x-axis
    ctxLoss.strokeStyle = '#100c09'
    ctxLoss.beginPath()
    ctxLoss.moveTo(padding, height - padding)
    ctxLoss.lineTo(width - padding, height - padding)
    ctxLoss.stroke()
    ctxLoss.closePath()

    // Draw y-axis
    ctxLoss.beginPath()
    ctxLoss.moveTo(padding, height - padding)
    ctxLoss.lineTo(padding, padding)
    ctxLoss.stroke()
    ctxLoss.closePath()

    // Add x-axis label
    ctxLoss.fillStyle = '#100c09'
    ctxLoss.font = '12px Arial'
    ctxLoss.fillText('Epochs', width / 2, height - padding / 4)

    // Add y-axis label
    ctxLoss.save()
    ctxLoss.translate(padding / 2, height / 2)
    ctxLoss.rotate(-Math.PI / 2)
    ctxLoss.textAlign = 'center'
    ctxLoss.fillText('Loss', 0, 0)
    ctxLoss.restore()
  }

  train() {
    const [xs, ys] = this.trainingData
    let epoch = 0
    let learningRate = 0.5
    const alpha = 1e-4
    const batch = xs.length

    const trainStep = () => {
      if (!this.training) return

      // Shuffle and get batch indices
      let indices = range(0, xs.length).sort(() => Math.random() - 0.5)
      if (batch) indices = indices.slice(0, batch)

      const xsb = indices.map((index) => xs[index])
      const ysb = indices.map((index) => ys[index])

      const yPredictions = xsb.map((x) => this.model.forward(x))
      const yTrue = ysb.map((label) => oneHotEncode(label, 3))
      const dataLoss = crossEntropyLoss(yPredictions, yTrue)

      const regLoss = new Value(alpha).mul(
        this.model
          .parameters()
          .reduce((acc, p) => acc.add(p.pow(2)), new Value(0)),
      )

      const totalLoss = dataLoss.add(regLoss)

      const correct = indices.filter(
        (index, i) => oneHotDecode(yPredictions[i]) === ys[index],
      ).length
      const accuracy = correct / yPredictions.length

      // Backward pass
      this.model.zeroGrad()
      totalLoss.backward()

      // update weights
      learningRate *= 0.99
      this.model.parameters().forEach((p) => {
        p.data -= learningRate * p.grad
      })

      this.model.epoch = epoch
      this.model.loss = totalLoss.data
      this.model.accuracy = accuracy
      this.lossHistory.push(totalLoss.data)

      console.info(
        `Epoch: ${epoch}, Loss: ${totalLoss.data.toFixed(
          5,
        )}, Accuracy: ${percent(
          accuracy,
        )}, Learning Rate: ${learningRate.toFixed(5)}`,
      )

      epoch += 1

      if (epoch > this.maxEpochs) return
      this.trainId = setTimeout(trainStep, 100)
    }

    trainStep()
  }

  setup() {
    this.model = buildModel()
    this.trainingData = generateLinearData(
      150,
      (x, y) => (y < -0.5 * x + 0.5 && 1) || (y > 0.75 && 2) || 0,
    )
    this.lossHistory = []
    this.training = true
    this.train()

    visualizeNetwork('.architecture', size, this.model.toString())
  }

  draw() {
    this.clear()
    this.plotBoundaries()
    this.plotPattern()
    this.plotLossCurve()

    const formatter = new Intl.NumberFormat('en-US', {
      style: 'decimal',
      maximumFractionDigits: 0,
    })

    document.querySelector('#epoch .value').innerText = formatter.format(
      this.model.epoch,
    )
    document.querySelector('#loss .value').innerText =
      this.model.loss.toFixed(5)
    document.querySelector('#accuracy .value').innerText = percent(
      this.model.accuracy,
    )
  }

  stop() {
    this.training = false
    clearTimeout(this.trainId)
    cancelAnimationFrame(this.drawId)
  }

  restart() {
    this.stop()
    this.start()
  }
}

const classifier = new MultiClassClassification({
  canvas: document.getElementById('plane'),
  lossCanvas: document.getElementById('lossCurve'),
  padding: 0,
  width: size,
  height: size,
})
classifier.start()
