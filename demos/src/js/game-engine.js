import range from './range.js'

/*
 * Partially ported from simple game engine by Andrej Karpathy
 * https://cs.stanford.edu/people/karpathy/convnetjs/demo/npgmain.js
 *
 * #setup()           called once in beginning
 * #update()          called every frame
 * #draw()            called every frame
 * #mouseClick(x, y)  callled on mouse click
 *
 * @TODO: Consider simplifying drawing with with p5.js as well
 */

export default class GameEngine {
  constructor({
    canvas,
    width = 500,
    height = 500,
    padding = 20,
    context = '2d',
    fps = 60,
  }) {
    this.canvas = canvas
    this.ctx = this.canvas.getContext(context)
    this.canvas.setAttribute('width', width)
    this.canvas.setAttribute('height', height)

    this.width = width
    this.height = height
    this.centerX = width / 2
    this.centerY = height / 2
    this.padding = padding
    this.innerWidth = this.width - 2 * this.padding
    this.innerHeight = this.height - 2 * this.padding
    this.drawId = null
    this.fps = fps
  }

  setup() {}

  update() {}

  draw() {}

  mouseClick(x, y) {}

  async start() {
    this.canvas.width = this.width
    this.canvas.height = this.height
    this.canvas.addEventListener('click', this.eventClick.bind(this), false)

    await this.setup()
    this.drawId = requestAnimationFrame(this.tick.bind(this))
  }

  tick() {
    const startTime = Date.now()

    this.update()
    this.draw()

    const endTime = Date.now()
    const deltaTime = endTime - startTime
    const waitTime = 1000 / this.fps - deltaTime
    if (waitTime > 0) {
      setTimeout(() => {
        this.drawId = requestAnimationFrame(this.tick.bind(this))
      }, waitTime)
    } else {
      this.drawId = requestAnimationFrame(this.tick.bind(this))
    }
  }

  eventClick(e) {
    // get position of cursor relative to top left of canvas
    let x
    let y
    if (e.pageX || e.pageY) {
      x = e.pageX
      y = e.pageY
    } else {
      x =
        e.clientX +
        document.body.scrollLeft +
        document.documentElement.scrollLeft
      y =
        e.clientY + document.body.scrollTop + document.documentElement.scrollTop
    }
    x -= this.canvas.offsetLeft
    y -= this.canvas.offsetTop

    this.mouseClick(x, y)
  }

  clear() {
    this.ctx.clearRect(0, 0, this.width, this.height)
  }

  drawRectangle(x, y, w, h, color) {
    this.ctx.beginPath()
    this.ctx.rect(x, y, w, h)
    this.ctx.closePath()
    this.ctx.fillStyle = color
    this.ctx.fill()
  }

  drawCircle(x, y, r, color) {
    this.ctx.beginPath()
    this.ctx.arc(x, y, r, 0, Math.PI * 2, true)
    this.ctx.closePath()
    this.ctx.fillStyle = color
    this.ctx.fill()
  }

  drawCoordinatePlan(xMin = -5, xMax = 5, yMin = -5, yMax = 5) {
    const padding = 20
    const innerWidth = this.width - 2 * padding
    const innerHeight = this.height - 2 * padding

    const mapX = (x) => padding + ((x - xMin) / (xMax - xMin)) * innerWidth
    const mapY = (y) =>
      this.height - padding - ((y - yMin) / (yMax - yMin)) * innerHeight

    this.ctx.lineWidth = 1
    this.ctx.setLineDash([])

    // Draw the x-axis
    this.ctx.beginPath()
    this.ctx.moveTo(padding, this.centerY)
    this.ctx.lineTo(this.width - padding, this.centerY)
    this.ctx.strokeStyle = '#100c09'
    this.ctx.stroke()

    // Draw the y-axis
    this.ctx.beginPath()
    this.ctx.moveTo(this.centerX, padding)
    this.ctx.lineTo(this.centerX, this.height - padding)
    this.ctx.strokeStyle = '#100c09'
    this.ctx.stroke()

    // Draw ticks and labels on x-axis
    range(xMin, xMax + 1).forEach((x, i) => {
      const xPos = mapX(x)
      this.ctx.beginPath()
      this.ctx.moveTo(xPos, this.centerY - 5)
      this.ctx.lineTo(xPos, this.centerY + 5)
      this.ctx.stroke()

      if (x !== 0 && !(i % 2)) {
        this.ctx.fillText(x, xPos - 3, this.centerY + 15)
      }
    })

    // Draw ticks and labels on y-axis
    range(yMin, yMax + 1).forEach((y, i) => {
      const yPos = mapY(y)
      this.ctx.beginPath()
      this.ctx.moveTo(this.centerX - 5, yPos)
      this.ctx.lineTo(this.centerX + 5, yPos)
      this.ctx.stroke()

      if (y !== 0 && !(i % 2)) {
        this.ctx.fillText(y, this.centerX + 10, yPos + 3)
      }
    })
  }
}
