import bootstrapModel from './bootstrap-model.js'

const weights =
  '[[{"weights":[-0.21632627741709443,2.946365950491937],"bias":-1.1096204228249589},{"weights":[0.042026533691509195,0.31283669585881685],"bias":-0.15101917761597777},{"weights":[0.13806609109826454,2.181174494255367],"bias":-0.9040039679479968},{"weights":[0.10778762131800752,1.4564344021987259],"bias":-0.5410020237990221}],[{"weights":[1.3761752398372382,0.11028477494474086,0.6925711353186949,0.5262157319612195],"bias":-0.3884697984705779},{"weights":[1.3358546166870016,0.14648278730809197,1.4079699288543395,1.0633108913566198],"bias":-0.043218859890338045},{"weights":[1.2599000646098704,0.18207837412895522,0.7981811219456962,0.23805385760073203],"bias":-1.360203704325321}],[{"weights":[-1.0705550203623766,-1.8838090224328763,-0.5265923334419728],"bias":2.844982386250495},{"weights":[0.3206482514648421,1.0781357752328493,-0.9852689565150748],"bias":-0.18285858085609727},{"weights":[1.1155764458948423,0.5978202589098995,1.539837284059666],"bias":-2.6621237841357157}]]'

const oneHotDecode = (values) => {
  const probs = values.map((v) => v.data)
  return probs.indexOf(Math.max(...probs))
}

describe('bootstrapModel', () => {
  it('should create a model with the correct structure and activations', () => {
    const structure = JSON.parse(weights)
    const model = bootstrapModel(structure)
    expect(model.layers).toHaveLength(3)
    expect(model.layers[0].activation).toBe('relu')
    expect(model.layers[1].activation).toBe('relu')
    expect(model.layers[2].activation).toBe('softmax')
  })

  it('should return valid forward pass', () => {
    const structure = JSON.parse(weights)
    const model = bootstrapModel(structure)
    expect(oneHotDecode(model.forward([0, 1]))).toEqual(2)
    expect(oneHotDecode(model.forward([0, 0.75]))).toEqual(1)
    expect(oneHotDecode(model.forward([0.75, 0.25]))).toEqual(0)
  })
})
