import { Value, utils } from '@andypai/neuroflow'

import GameEngine from './js/game-engine.js'
import range from './js/range.js'
import buildModel from './js/mnist-model.js'
import visualizeNetwork from './js/network-visualization.js'

const size = Math.min(+document.querySelector('.train').offsetWidth, 600)

const { oneHot, crossEntropyLoss } = utils

const percent = (v) =>
  new Intl.NumberFormat('en-US', { style: 'percent' }).format(v)

class MNISTClassification extends GameEngine {
  constructor({ canvas, lossCanvas, trainingData, testData }) {
    super({ canvas })
    this.lossCanvas = lossCanvas
    this.ctxLoss = this.lossCanvas.getContext('2d')
    this.lossCanvas.setAttribute('width', size)
    this.lossCanvas.setAttribute('height', 200)

    this.model = buildModel()
    this.training = false
    this.trainId = null
    this.drawId = null
    this.maxEpochs = 100
    this.trainingData = trainingData
    this.trainingBatch = []
    this.testData = testData
    this.lossHistory = []

    this.size = 14
    this.scale = this.size * 4
    this.fps = 5
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
    ctxLoss.strokeStyle = 'rgb(202 15 81 / 18%)'
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

    ctxLoss.strokeStyle = 'rgb(202 15 81 / 100%)' // Different color for moving average
    ctxLoss.beginPath()

    const movingAveragePeriod = 10
    const lossHistoryAverage = this.lossHistory.map((val, idx, arr) => {
      const start = Math.max(0, idx - movingAveragePeriod + 1)
      const subset = arr.slice(start, idx + 1)
      return subset.reduce((a, b) => a + b, 0) / subset.length
    })

    lossHistoryAverage.forEach((avgLoss, index) => {
      const x = padding + (index / (this.maxEpochs - 1)) * plotWidth
      const y = height - padding - (avgLoss / maxLoss) * plotHeight

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

  plotBatch() {
    const type = 'train'
    const selector = `.${type} .train-grid`
    document.querySelector(selector).innerHTML = ''

    const canvas = document.createElement('canvas')
    canvas.id = 'combinedCanvas'
    const imagesPerRow = 4 // Adjust this value as needed
    const numImages = 16
    canvas.width = this.scale * imagesPerRow
    canvas.height = this.scale * Math.ceil(numImages / imagesPerRow)
    document.querySelector(selector).appendChild(canvas)

    const [xs] = this.trainingBatch
    xs.slice(0, 20).forEach((image, index) => {
      const x = index % imagesPerRow
      const y = Math.floor(index / imagesPerRow)
      this.drawImage({
        canvas,
        image,
        xPos: x * this.scale,
        yPos: y * this.scale,
      })
    })
  }

  plotTest() {
    if (this.model.epoch % 10 !== 0) return

    let totalCorrect = 0
    const incorrect = []
    const [xs, ys] = this.testData

    const type = 'test'
    const selector = `.${type} .test-grid`
    document.querySelector(selector).innerHTML = ''

    const padding = 5 // Adjust padding as needed
    const imagesPerRow = 5 // Adjust this value as needed
    const numImages = xs.length // Adjust this value as needed
    const canvas = document.createElement('canvas')
    canvas.id = 'combinedCanvas'
    canvas.width = this.scale * imagesPerRow + padding * (imagesPerRow + 1)
    canvas.height =
      this.scale * Math.ceil(numImages / imagesPerRow) +
      padding * (Math.ceil(numImages / imagesPerRow) + 1)
    document.querySelector(selector).appendChild(canvas)

    xs.forEach((x, index) => {
      const prediction = this.model.forward(x).map((v) => v.data)
      const label = oneHot.decode(prediction)
      const correct = label === ys[index]
      if (correct) totalCorrect += 1
      if (!correct) incorrect.push({ guess: label, answer: ys[index], index })
      const xiPos = index % imagesPerRow
      const yiPos = Math.floor(index / imagesPerRow)
      const xPos = xiPos * this.scale + padding * (xiPos + 1)
      const yPos = yiPos * this.scale + padding * (yiPos + 1)
      const { ctx } = this.drawImage({ canvas, image: x, xPos, yPos })

      // Draw border
      ctx.strokeStyle = correct ? 'green' : 'red'
      ctx.lineWidth = 2
      ctx.strokeRect(xPos, yPos, this.scale, this.scale)
    })

    document.querySelector('.test .summary .epoch').innerText = this.model.epoch
    document.querySelector('.test .summary .correct').innerText = totalCorrect
    document.querySelector('.test .summary .total').innerText = xs.length
    document.querySelector('.test .summary .accuracy').innerText = percent(
      totalCorrect / xs.length,
    )

    // Create a table of incorrect guesses
    const tableHtml = this.lossTable(incorrect, xs)
    document.querySelector('.test-results .table').innerHTML = ''
    document.querySelector('.test-results .table').innerHTML = tableHtml
  }

  drawImage({ canvas, image, xPos = 0, yPos = 0 }) {
    let imageCanvas = canvas

    if (!canvas) {
      imageCanvas = document.createElement('canvas')
      imageCanvas.width = this.scale
      imageCanvas.height = this.scale
    }

    const ctx = imageCanvas.getContext('2d')
    const imageData = ctx.createImageData(this.size, this.size)
    image.forEach((pixel, idx) => {
      const value = pixel * 255
      imageData.data[idx * 4 + 0] = value // Red
      imageData.data[idx * 4 + 1] = value // Green
      imageData.data[idx * 4 + 2] = value // Blue
      imageData.data[idx * 4 + 3] = 255 // Alpha
    })

    // Create a temporary canvas to scale the image
    const tempCanvas = document.createElement('canvas')
    tempCanvas.width = this.size
    tempCanvas.height = this.size
    const tempCtx = tempCanvas.getContext('2d')
    tempCtx.putImageData(imageData, 0, 0)

    // Scale up the image and draw it on the image canvas
    ctx.imageSmoothingEnabled = false // Disable smoothing for pixelated effect
    ctx.drawImage(
      tempCanvas,
      0,
      0,
      this.size,
      this.size,
      xPos,
      yPos,
      this.scale,
      this.scale,
    )

    return {
      canvas: imageCanvas,
      ctx,
    }
  }

  lossTable(incorrect, xs) {
    let tableHtml =
      '<table><thead><tr><th>Image</th><th>Guess</th><th>Answer</th></tr></thead><tbody>'
    incorrect.forEach(({ guess, answer, index }) => {
      const { canvas } = this.drawImage({ image: xs[index] })
      const imageHtml = `<img src="${canvas.toDataURL()}" width="${
        this.scale
      }" height="${this.scale}">`
      tableHtml += `<tr><td>${imageHtml}</td><td>${guess}</td><td>${answer}</td></tr>`
    })
    tableHtml += '</tbody></table>'
    return tableHtml
  }

  train() {
    const [xs, ys] = this.trainingData

    let epoch = 0
    let learningRate = 0.01
    const alpha = 1e-3
    const batchSize = 16
    const lambda = 0.01
    const uniqueLabels = 10

    const trainStep = () => {
      if (!this.training) return

      let loss
      const batchIdx = range(0, xs.length)
        .sort(() => Math.random() - 0.5)
        .slice(0, batchSize)

      const batchXs = batchIdx.map((idx) => xs[idx])
      const batchYs = batchIdx.map((idx) => ys[idx])
      this.trainingBatch = [batchXs, batchYs]

      loss = new Value(0)
      const predictions = []

      batchXs.forEach((x, j) => {
        const prediction = this.model.forward(x)
        predictions.push(prediction)
        const label = oneHot.encode(batchYs[j], uniqueLabels)
        loss = loss.add(crossEntropyLoss(prediction, label))
      })

      // L2 Regularization term
      const l2Loss = this.model
        .parameters()
        .reduce((acc, param) => acc.add(param.mul(param)), new Value(0))
        .mul(lambda / 2)

      // Add L2 regularization term to the loss
      loss = loss.add(l2Loss).div(batchSize)
      loss.backward()

      // Update model parameters in the opposite direction of the gradient
      this.model.parameters().forEach((param) => {
        param.data -= learningRate * param.grad
        param.grad = 0
      })

      // Adjust learning rate
      learningRate = +(learningRate * (1 - alpha)).toFixed(10) || 0.00000001

      const accuracy =
        predictions.reduce((acc, pred, j) => {
          const label = batchYs[j]
          const prediction = oneHot.decode(pred.map((v) => v.data))
          return acc + (label === prediction ? 1 : 0)
        }, 0) / batchSize

      this.model.epoch = epoch
      this.model.loss = loss.data
      this.model.accuracy = accuracy
      this.model.learningRate = learningRate
      this.model.alpha = alpha
      this.model.batchSize = batchSize
      this.model.lambda = lambda
      this.lossHistory.push(loss.data)

      console.info(
        `Epoch: ${epoch}, Loss: ${loss.data.toFixed(5)}, Accuracy: ${percent(
          accuracy,
        )}, Learning Rate: ${learningRate.toFixed(5)}`,
      )

      epoch += 1

      if (epoch > this.maxEpochs) return
      this.trainId = setTimeout(trainStep, 500)
    }

    trainStep()
  }

  setup() {
    this.model = buildModel()
    this.lossHistory = []
    this.training = true
    this.train()

    visualizeNetwork('.architecture', size, this.model.toString())
  }

  async draw() {
    this.clear()
    this.plotBatch()
    this.plotTest()
    this.plotLossCurve()

    const formatter = new Intl.NumberFormat('en-US', {
      style: 'decimal',
      maximumFractionDigits: 0,
    })

    document.querySelector('#epoch .value').innerText = formatter.format(
      this.model.epoch,
    )
    document.querySelector('#loss .value').innerText =
      this.model.loss?.toFixed(5)
    document.querySelector('#accuracy .value').innerText = percent(
      this.model.accuracy,
    )
    document.querySelector('#learning-rate .value').innerText =
      this.model.learningRate
    document.querySelector('#batch-size .value').innerText =
      this.model.batchSize
    document.querySelector('#alpha .value').innerText = this.model.alpha
    document.querySelector('#lambda .value').innerText = this.model.lambda
    await new Promise((resolve) => {
      setTimeout(resolve, 100000)
    })
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

const fetchTrainingData = async () => {
  const response = await fetch(`${process.env.API_URL}/mnist?end=1000`)
  const json = await response.json()
  return json
}

const fetchTestData = async () => {
  const response = await fetch(`${process.env.API_URL}/mnist?end=30&type=test`)
  const json = await response.json()
  return json
}

const main = async () => {
  const [train, test] = await Promise.all([
    fetchTrainingData(),
    fetchTestData(),
  ])

  const trainingData = train.reduce(
    (prev, data) => {
      const { image, label } = data
      prev[0].push(image)
      prev[1].push(label)
      return prev
    },
    [[], []],
  )

  const testData = test.reduce(
    (prev, data) => {
      const { image, label } = data
      prev[0].push(image)
      prev[1].push(label)
      return prev
    },
    [[], []],
  )

  const classifier = new MNISTClassification({
    canvas: document.createElement('canvas'),
    lossCanvas: document.getElementById('lossCurve'),
    padding: 0,
    width: size,
    height: size,
    trainingData,
    testData,
  })

  classifier.start()
}

main()
