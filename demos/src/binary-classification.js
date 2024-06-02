import { Value } from '@andypai/neuroflow'

import GameEngine from './js/game-engine.js'
import patterns from './js/patterns.js'
import range from './js/range.js'
import buildModel from './js/binary-classification-model.js'
import visualizeNetwork from './js/network-visualization.js'

const size = Math.min(+document.querySelector('.demo').offsetWidth, 600)

const percent = (v) =>
  new Intl.NumberFormat('en-US', { style: 'percent' }).format(v)

class BinaryClassification extends GameEngine {
  constructor({ canvas, lossCanvas }) {
    super({ canvas })
    this.pattern = 'circles'

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

  getPattern() {
    return this.pattern
  }

  setPattern(pattern) {
    this.pattern = pattern
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
        (label === 1 && '#3a7ca5') || (label === -1 && '#f6511d') || '#006632'
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
        const label = prediction.data > 0 ? 1 : -1
        const color =
          (label === 1 && 'rgb(58 124 165 / 20%)') ||
          (label === -1 && 'rgb(246 81 29 / 20%)') ||
          'rgb(0 102 50 / 20%)'
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
    let learningRate = 0.8
    const alpha = 1e-4
    const batch = xs.length

    const trainStep = () => {
      if (!this.training) return

      // Shuffle and get batch indices
      let indices = range(0, xs.length).sort(() => Math.random() - 0.5)
      if (batch) indices = indices.slice(0, batch)

      const yPredictions = indices.map((index) => this.model.forward(xs[index]))

      const losses = yPredictions.reduce((acc, yPrediction, index) => {
        const loss = yPrediction.mul(-ys[indices[index]]).add(1).relu()
        return loss.add(acc)
      }, new Value(0))

      const dataLoss = losses.div(yPredictions.length)

      const regLoss = new Value(alpha).mul(
        this.model
          .parameters()
          .reduce((acc, p) => acc.add(p.pow(2)), new Value(0)),
      )

      const totalLoss = dataLoss.add(regLoss)

      const correct = yPredictions.filter(
        (y, index) => y.data > 0 === ys[indices[index]] > 0,
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
    const pattern = this.getPattern()

    console.info(`New model for expression: ${pattern}`)
    this.model = buildModel()
    this.trainingData = patterns(pattern, 100)
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

const classifier = new BinaryClassification({
  canvas: document.getElementById('plane'),
  lossCanvas: document.getElementById('lossCurve'),
  padding: 0,
  width: size,
  height: size,
})
classifier.start()

const reload = (pattern) => {
  classifier.setPattern(pattern)
  classifier.restart()
}

const buttons = document.querySelectorAll('.types button')
buttons.forEach((button) => {
  button.addEventListener('click', () => {
    const pattern = button.getAttribute('data-pattern')
    reload(pattern)
  })
})
