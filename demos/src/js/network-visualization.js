import P5 from 'p5'

const inputNeuronColor = '#3a7ca5'
const reluNeuronColor = '#f6511d'
const outputNeuronColor = '#006632'

const parseStructure = (structure) => {
  const layers = structure.match(/Layer of \[([^\]]+)\]/g).map((layer) => {
    const neurons = layer.match(/(RELUNeuron|LINEARNeuron)\(\d+\)/g)
    return neurons.map((neuron) => {
      const type = neuron.match(/(RELUNeuron|LINEARNeuron)/)[0]
      const count = +neuron.match(/\d+/)[0]
      return { type, count }
    })
  })
  return layers
}

const addInputLayer = (layers) => {
  const inputLayerSize = layers[0][0].count // Number inside the neuron of the first layer
  const inputLayer = Array(inputLayerSize).fill({
    type: 'INPUTNeuron',
    count: 1,
  })
  layers.unshift(inputLayer)
}

const drawNeuron = (p5, x, y, type) => {
  p5.stroke(0)
  if (type === 'RELUNeuron') {
    p5.fill(reluNeuronColor)
  } else if (type === 'LINEARNeuron') {
    p5.fill(outputNeuronColor)
  } else if (type === 'INPUTNeuron') {
    p5.fill(inputNeuronColor)
  }
  p5.ellipse(x, y, 20, 20)
}

const drawConnection = (p5, x1, y1, x2, y2) => {
  p5.stroke(200)
  p5.line(x1, y1, x2, y2)
}

const drawLegend = (p5) => {
  const legendX = 20
  const legendY = 20
  const legendSpacing = 25

  p5.fill('#3a7ca5')
  p5.ellipse(legendX, legendY, 20, 20)
  p5.fill(0)
  p5.text('INPUTNeuron', legendX + 30, legendY + 5)

  p5.fill('#f6511d')
  p5.ellipse(legendX, legendY + legendSpacing, 20, 20)
  p5.fill(0)
  p5.text('RELUNeuron', legendX + 30, legendY + legendSpacing + 5)

  p5.fill('#006632')
  p5.ellipse(legendX, legendY + legendSpacing * 2, 20, 20)
  p5.fill(0)
  p5.text('LINEARNeuron', legendX + 30, legendY + legendSpacing * 2 + 5)
}

const drawNetwork = (p5, layers) => {
  const layerHeight = Math.min(p5.height / layers.length, 100)
  layers.forEach((layer, layerIndex) => {
    const layerWidth = p5.width * 0.9
    const xOffset = (p5.width - layerWidth) / 2
    const yOffset = layerHeight * layerIndex + layerHeight / 2
    const neuronSpacing = layerWidth / (layer.length + 1)
    layer.forEach((neuron, neuronIndex) => {
      const neuronX = xOffset + neuronSpacing * (neuronIndex + 1)
      const neuronY = yOffset
      drawNeuron(p5, neuronX, neuronY, neuron.type)
      if (layerIndex > 0) {
        const previousLayer = layers[layerIndex - 1]
        const previousLayerNeuronSpacing =
          layerWidth / (previousLayer.length + 1)
        previousLayer.forEach((prevNeuron, prevNeuronIndex) => {
          const prevNeuronX =
            xOffset + previousLayerNeuronSpacing * (prevNeuronIndex + 1)
          const prevNeuronY = yOffset - layerHeight
          drawConnection(p5, prevNeuronX, prevNeuronY, neuronX, neuronY)
        })
      }
    })
  })
  drawLegend(p5)
}

const removeExistingCanvas = (parent) => {
  const parentElement = document.querySelector(parent)
  if (parentElement) {
    const existingCanvas = parentElement.querySelector('canvas')
    if (existingCanvas) {
      parentElement.removeChild(existingCanvas)
    }
  }
}
const windowResized = (p5, size, structure) => {
  p5.resizeCanvas(size, size)
  const layers = parseStructure(structure)
  addInputLayer(layers)
  drawNetwork(p5, layers)
}

const setup = (p5, parent, size, structure) => {
  removeExistingCanvas(parent)

  const canvas = p5.createCanvas(size, size)
  canvas.parent(p5.select(parent))
  const layers = parseStructure(structure)
  addInputLayer(layers)
  drawNetwork(p5, layers)
}

export default (parent, size, structure) =>
  new P5((p) => {
    p.setup = () => setup(p, parent, size, structure)
    p.windowResized = () => windowResized(p, size, structure)
  })
