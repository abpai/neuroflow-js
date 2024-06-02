export default class Value {
  // stores a single scalar value and its gradient
  constructor(data, _children = [], _op = '') {
    this.data = data
    this.grad = 0
    // internal variables used for autograd graph construction
    this._backward = () => {}
    this._prev = new Set(_children)
    this._op = _op // the op that produced this node, for graphviz / debugging / etc
  }

  static ensureValue(other) {
    return other instanceof Value ? other : new Value(other)
  }

  neg() {
    return this.mul(-1)
  }

  add(_other) {
    const other = Value.ensureValue(_other)
    const out = new Value(this.data + other.data, [this, other], '+')

    const _backward = () => {
      this.grad += out.grad
      other.grad += out.grad
    }
    out._backward = _backward

    return out
  }

  sub(_other) {
    const other = Value.ensureValue(_other)
    return this.add(other.neg())
  }

  mul(_other) {
    const other = Value.ensureValue(_other)
    const out = new Value(this.data * other.data, [this, other], '*')

    const _backward = () => {
      this.grad += other.data * out.grad
      other.grad += this.data * out.grad
    }
    out._backward = _backward

    return out
  }

  div(_other) {
    const other = Value.ensureValue(_other)
    return this.mul(other.pow(-1))
  }

  exp() {
    const out = new Value(Math.exp(this.data), [this], 'exp')

    const _backward = () => {
      this.grad += out.data * out.grad
    }
    out._backward = _backward

    return out
  }

  pow(other) {
    if (typeof other !== 'number') {
      throw new Error('only supporting int/float powers for now')
    }
    const out = new Value(this.data ** other, [this], `**${other}`)

    const _backward = () => {
      this.grad += other * this.data ** (other - 1) * out.grad
    }
    out._backward = _backward

    return out
  }

  log() {
    const out = new Value(Math.log(this.data), [this], 'log')

    const _backward = () => {
      this.grad += (1 / this.data) * out.grad
    }
    out._backward = _backward

    return out
  }

  relu() {
    const out = new Value(this.data < 0 ? 0 : this.data, [this], 'ReLU')

    const _backward = () => {
      this.grad = (out.data > 0) * out.grad
    }
    out._backward = _backward

    return out
  }

  tanh() {
    const t = (Math.exp(2 * this.data) - 1) / (Math.exp(2 * this.data) + 1)
    const out = new Value(t, [this], 'tanh')
    const _backward = () => {
      this.grad = (1 - t ** 2) * out.grad
    }
    out._backward = _backward

    return out
  }

  static softmax(values) {
    const expValues = values.map((val) => val.exp())
    const sumExpValues = expValues.reduce((a, b) => a.add(b), new Value(0))
    const outValues = expValues.map((expVal, i) => {
      const out = expVal.div(sumExpValues)
      const _backward = () => {
        const softmaxVal = out.data
        values.forEach((val, j) => {
          if (i === j) {
            val.grad += softmaxVal * (1 - softmaxVal) * out.grad
          } else {
            val.grad +=
              -softmaxVal * (expValues[j].data / sumExpValues.data) * out.grad
          }
        })
      }
      out._backward = _backward
      return out
    })
    return outValues
  }

  backward() {
    // topological order all of the children in the graph
    const topo = []
    const visited = new Set()
    const buildTopo = (v) => {
      if (!visited.has(v)) {
        visited.add(v)
        v._prev.forEach((child) => buildTopo(child))
        topo.push(v)
      }
    }
    buildTopo(this)

    // go one variable at a time and apply the chain rule to get its gradient
    this.grad = 1
    topo.reverse().forEach((v) => v._backward())
  }

  toString() {
    return this._op
      ? `Value(data=${this.data}, grad=${this.grad}, op=${this._op})`
      : `Value(data=${this.data}, grad=${this.grad})`
  }
}
