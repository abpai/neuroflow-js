export default class Value {
  // stores a single scalar value and its gradient
  constructor(data, _children = [], _op = '') {
    this.data = Value.limitPrecision(data)
    this.grad = 0
    // internal variables used for autograd graph construction
    this._backward = () => {}
    this._prev = new Set(_children)
    this._op = _op // the op that produced this node, for graphviz / debugging / etc
  }

  static limitPrecision(value) {
    return +value.toFixed(8)
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

    out._backward = () => {
      this.grad += out.grad
      other.grad += out.grad
    }

    return out
  }

  sub(_other) {
    const other = Value.ensureValue(_other)
    return this.add(other.neg())
  }

  mul(_other) {
    const other = Value.ensureValue(_other)
    const out = new Value(this.data * other.data, [this, other], '*')

    out._backward = () => {
      this.grad += other.data * out.grad
      other.grad += this.data * out.grad
    }

    return out
  }

  div(_other) {
    const other = Value.ensureValue(_other)
    return this.mul(other.pow(-1))
  }

  exp() {
    const out = new Value(Math.exp(this.data), [this], 'exp')

    out._backward = () => {
      this.grad += out.data * out.grad
    }

    return out
  }

  pow(other) {
    if (typeof other !== 'number') {
      throw new Error('only supporting int/float powers for now')
    }
    const out = new Value(this.data ** other, [this], `**${other}`)

    out._backward = () => {
      this.grad += other * this.data ** (other - 1) * out.grad
    }

    return out
  }

  log(epsilon = 1e-8) {
    if (!this.data) this.data = epsilon

    const out = new Value(Math.log(this.data), [this], 'log')
    out._backward = () => {
      this.grad += (1 / this.data) * out.grad
    }

    return out
  }

  relu() {
    const out = new Value(this.data < 0 ? 0 : this.data, [this], 'ReLU')

    out._backward = () => {
      this.grad = (out.data > 0) * out.grad
    }

    return out
  }

  tanh() {
    const t = (Math.exp(2 * this.data) - 1) / (Math.exp(2 * this.data) + 1)
    const out = new Value(t, [this], 'tanh')
    out._backward = () => {
      this.grad = (1 - t ** 2) * out.grad
    }

    return out
  }

  static softmax(values) {
    const expValues = values.map((val) => val.exp())
    const sumExpValues = expValues.reduce((a, b) => a.add(b), new Value(0))
    const outValues = expValues.map((expVal, i) => {
      const out = expVal.div(sumExpValues)
      out._backward = () => {
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
      return out
    })
    return outValues
  }

  // topological order all of the children in the graph
  // doesn't use recursion to avoid max call stack size exceeded errors
  backward() {
    const topo = [] // List to store nodes in topological order
    const visited = new Set() // Set to track visited nodes
    const addedToTopo = new Set() // Set to ensure nodes are only added to topo once
    const stack = [this] // Stack to manage the iterative DFS

    // Build the topological order using an iterative approach
    while (stack.length > 0) {
      const node = stack[stack.length - 1] // Peek at the top node of the stack
      if (!visited.has(node)) {
        visited.add(node)
        let allChildrenVisited = true
        // Iterate over node._prev in reverse order to preserve the correct processing order
        Array.from(node._prev)
          .reverse()
          .forEach((child) => {
            if (!visited.has(child)) {
              stack.push(child) // Push unvisited children onto the stack
              allChildrenVisited = false
            }
          })
        if (allChildrenVisited) {
          stack.pop() // All children are visited, so remove the node from the stack
          if (!addedToTopo.has(node)) {
            // Check if the node is already added to topo
            topo.push(node) // Add the node to the topological order
            addedToTopo.add(node) // Mark the node as added
          }
        }
      } else {
        stack.pop() // Node has been visited, remove it from the stack
        if (!addedToTopo.has(node)) {
          // Ensure the node is only added once
          topo.push(node) // Add the node to the topological order
          addedToTopo.add(node) // Mark the node as added
        }
      }
    }

    // Reverse the topological order and apply the chain rule
    this.grad = 1 // Initialize the gradient of the output node
    topo.reverse().forEach((v) => v._backward()) // Apply backward pass in topological order
  }

  toString() {
    return this._op
      ? `Value(data=${this.data}, grad=${this.grad}, op=${this._op})`
      : `Value(data=${this.data}, grad=${this.grad})`
  }
}
