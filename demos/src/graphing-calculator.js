import { evaluate } from 'mathjs'
import GameEngine from './js/game-engine.js'
import range from './js/range.js'
import { generateTrainingData, buildModel } from './js/regression-model.js'
import visualizeNetwork from './js/network-visualization.js'

const size = Math.min(+document.querySelector('.demo').offsetWidth, 600)

class GraphingCalculator extends GameEngine {
  constructor({
    canvas,
    lossCanvas,
    expression = 'sin(x)',
    xRange = { min: -5, max: 5 },
    yRange = { min: -5, max: 5 },
  }) {
    super({ canvas })
    this.expression = expression
    this.xRange = xRange
    this.yRange = yRange

    this.model = buildModel()
    this.training = false
    this.trainId = null
    this.drawId = null
    this.maxEpochs = 100000
    this.trainingData = []
    this.lossHistory = []

    this.lossCanvas = lossCanvas
    this.ctxLoss = this.lossCanvas.getContext('2d')
    this.lossCanvas.setAttribute('width', size)
    this.lossCanvas.setAttribute('height', 200)
  }

  getExpression() {
    return this.expression
  }

  setExpression(newExpression) {
    this.expression = newExpression
  }

  mapX(x) {
    return (
      this.padding +
      ((x - this.xRange.min) / (this.xRange.max - this.xRange.min)) *
        this.innerWidth
    )
  }

  mapY(y) {
    return (
      this.height -
      this.padding -
      ((y - this.yRange.min) / (this.yRange.max - this.yRange.min)) *
        this.innerHeight
    )
  }

  plotFunction(fn, { color = '#f6511d', lineWidth = 1, lineDash = [] } = {}) {
    const expression = this.getExpression()
    if (!fn) fn = (x) => evaluate(expression, { x })

    this.ctx.lineWidth = lineWidth
    this.ctx.strokeStyle = color
    this.ctx.setLineDash(lineDash)
    this.ctx.beginPath()

    range(this.xRange.min, this.xRange.max + 1, 0.1).forEach((x, i) => {
      const xPos = this.mapX(x)
      const yPos = this.mapY(fn(x))

      if (i === 0) {
        this.ctx.moveTo(xPos, yPos)
      } else {
        this.ctx.lineTo(xPos, yPos)
      }
    })

    this.ctx.stroke()
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
    let learningRate = 0.001
    let step = 0
    const decaySteps = 100
    const decayRate = 0.99
    const [xs, ys] = this.trainingData

    const trainStep = () => {
      if (!this.training) return

      // forward pass
      const indices = range(0, xs.length).sort(() => Math.random() - 0.5)

      indices.forEach((j) => {
        const yPrediction = this.model.forward(xs[j])
        const loss = yPrediction.sub(ys[j]).pow(2)

        // backward pass
        this.model.zeroGrad()
        loss.backward()

        // update weights
        this.model.parameters().forEach((p) => {
          p.data -= learningRate * p.grad
        })

        this.model.epoch = step * xs.length + j
        this.model.loss = loss.data
        this.lossHistory.push(loss.data)
      })

      if (step % decaySteps === 0) {
        learningRate *= decayRate
        console.info(`Updated learning rate: ${learningRate}`)
      }

      step += 1
      this.trainId = setTimeout(trainStep, 100)
    }

    trainStep()
  }

  setup() {
    const func = document.getElementById('fnInput')
    if (!func.value) func.value = this.getExpression()

    this.setExpression(func.value)

    console.info(`New model for expression: ${func.value}`)
    this.model = buildModel()
    this.trainingData = generateTrainingData(this.xRange, (x) =>
      evaluate(func.value, { x }),
    )
    this.training = true
    this.train()

    visualizeNetwork('.architecture', size, this.model.toString())
  }

  draw() {
    this.clear()
    this.drawCoordinatePlan(
      this.xRange.min,
      this.xRange.max,
      this.yRange.min,
      this.yRange.max,
    )
    this.plotFunction((x) => evaluate(this.getExpression(), { x }), {
      color: '#f6511d',
      lineWidth: 3,
    })
    this.plotFunction((x) => this.model.forward([x]).data, {
      color: '#006632',
      lineWidth: 3,
      lineDash: [10, 5],
    })
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

const classifier = new GraphingCalculator({
  canvas: document.getElementById('plane'),
  lossCanvas: document.getElementById('lossCurve'),
  width: size,
  height: size,
})
classifier.start()

const reload = (e) => {
  if (e) e.preventDefault()

  classifier.setExpression(document.getElementById('fnInput').value)
  classifier.restart()
}

document.querySelector('.fn').addEventListener('submit', reload)
document.getElementById('plot').addEventListener('click', reload)
