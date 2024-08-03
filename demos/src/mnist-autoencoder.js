import visualizeNetwork from './js/network-visualization.js'
import buildModel from './js/mnist-autoencoder-model.js'

const fetchTestData = async () => {
  const response = await fetch(`${process.env.API_URL}/mnist?end=50&type=test`)
  const json = await response.json()
  return json
}

const drawImage = (image) => {
  const imageCanvas = document.createElement('canvas')

  const size = 14
  const scale = size * 8

  imageCanvas.width = scale
  imageCanvas.height = scale

  const ctx = imageCanvas.getContext('2d')
  const imageData = ctx.createImageData(size, size)
  image.forEach((pixel, idx) => {
    const value = pixel * 255
    imageData.data[idx * 4 + 0] = value
    imageData.data[idx * 4 + 1] = value
    imageData.data[idx * 4 + 2] = value
    imageData.data[idx * 4 + 3] = 255
  })

  // Create a temporary canvas to scale the image
  const tempCanvas = document.createElement('canvas')
  tempCanvas.width = size
  tempCanvas.height = size
  const tempCtx = tempCanvas.getContext('2d')
  tempCtx.putImageData(imageData, 0, 0)

  // Scale up the image and draw it on the image canvas
  ctx.imageSmoothingEnabled = false
  ctx.drawImage(tempCanvas, 0, 0, size, size, 0, 0, scale, scale)
  return imageCanvas
}

let testData = null
let currentIteration = null

const getSelectedActivation = () =>
  document.querySelector('input[name="activation"]:checked').value

const iterateImages = (id) => async (autoencoder) => {
  if (!testData) testData = await fetchTestData()

  for (const { image: original } of testData) {
    if (currentIteration !== id) {
      break
    }

    const decoded = autoencoder.forward(original)
    const reconstructed = decoded.map(({ data: pixel }) =>
      pixel > 0.5 ? 1 : 0,
    )
    document.getElementById('original').innerHTML = ''
    document.getElementById('reconstructed').innerHTML = ''
    document.getElementById('original').appendChild(drawImage(original))
    document
      .getElementById('reconstructed')
      .appendChild(drawImage(reconstructed))
    await new Promise((resolve) => {
      setTimeout(resolve, 2000)
    })
  }
}

const main = async () => {
  const activation = getSelectedActivation()
  const autoencoder = await buildModel(activation)

  currentIteration = Math.random()
  iterateImages(currentIteration)(autoencoder)

  const size = Math.min(+document.querySelector('.demo').offsetWidth, 600)
  visualizeNetwork('#architecture', size, autoencoder.toString())
}

document.querySelectorAll('input[name="activation"]').forEach((radio) => {
  radio.addEventListener('change', () => {
    if (currentIteration) currentIteration = null
    main()
  })
})

main()
