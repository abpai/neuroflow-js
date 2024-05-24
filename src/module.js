export default class Module {
  // Resets the gradients of all parameters to zero
  zeroGrad() {
    this.parameters().forEach((param) => {
      param.grad = 0
    })
  }

  // eslint-disable-next-line class-methods-use-this
  parameters() {
    // Returns an empty list; meant to be overridden by subclasses
    return []
  }
}
