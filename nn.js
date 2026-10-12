'use strict';
/*
 * nn.js — the neural network, built from individual neurons and links.
 *
 * Same idea as the first sentiment.js: no matrices, no libraries.
 * Every neuron is an object, every link between two neurons is an object
 * with its own weight.
 *
 * What is new compared to the first version:
 *
 *  1. Sparse input. A review uses ~100 of the 5000 vocabulary words, so
 *     ~4900 input neurons output 0. A link from a 0 neuron adds 0 × weight = 0
 *     in the forward pass and gets an update of lr × delta × 0 = 0 in the
 *     backward pass. So we skip those links. The math is identical, it is
 *     just ~50× faster.
 *
 *  2. forward(), backward() and update() are separate methods, so the
 *     trainer can stop between them and write down what the network did.
 *
 *  3. Seeded random numbers. Same seed => same starting weights => the same
 *     run every time, which makes the logs comparable.
 *
 *  4. Checkpoints: the trained "knowledge" is just the list of weights and
 *     biases, so saving = writing those numbers, loading = putting them back.
 */

// Small deterministic random generator (mulberry32). Returns numbers in [0, 1).
function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const sigmoid = x => 1 / (1 + Math.exp(-x));

// One wire between two neurons. Holds exactly one learned number.
class Link {
  constructor(id, from, to, weight) {
    this.id = id;
    this.from = from;
    this.to = to;
    this.weight = weight;
  }
}

class Neuron {
  constructor(name, layerIndex, index) {
    this.name = name;            // e.g. "hidden1[3]"
    this.label = name;           // input neurons get the word they stand for, e.g. "great"
    this.layerIndex = layerIndex;
    this.index = index;
    this.incoming = [];
    this.outgoing = [];
    this.bias = 0;               // learned
    this.sum = 0;                // z = bias + Σ(input × weight), recomputed every forward pass
    this.output = 0;             // a = sigmoid(z)
    this.delta = 0;              // blame for the error, computed in backward()
  }

  // Dense forward step: read every incoming link.
  fire() {
    let total = this.bias;
    for (const link of this.incoming) total += link.from.output * link.weight;
    this.sum = total;
    this.output = sigmoid(total);
  }
}

class Layer {
  constructor(name, index, size) {
    this.name = name;
    this.index = index;
    this.neurons = Array.from({ length: size }, (_, i) => new Neuron(`${name}[${i}]`, index, i));
  }
}

class Network {
  // sizes: [5000, 16, 8, 1] = 5000 inputs, hidden layers of 16 and 8, 1 output
  constructor(sizes, { seed = 1, inputLabels = null, initScale = 0.5 } = {}) {
    this.sizes = sizes.slice();
    const rng = mulberry32(seed);

    this.layers = sizes.map((size, i) => {
      const name = i === 0 ? 'input' : i === sizes.length - 1 ? 'output' : `hidden${i}`;
      return new Layer(name, i, size);
    });
    if (inputLabels) this.layers[0].neurons.forEach((n, i) => { n.label = inputLabels[i]; });

    // Fully connect each layer to the next. Links are created in a fixed
    // order, so link #i is always the same wire, which is what lets a saved
    // list of weights be loaded back onto the right wires.
    this.links = [];
    for (let l = 0; l < this.layers.length - 1; l++) {
      for (const a of this.layers[l].neurons) {
        for (const b of this.layers[l + 1].neurons) {
          const link = new Link(this.links.length, a, b, (rng() * 2 - 1) * initScale);
          a.outgoing.push(link);
          b.incoming.push(link);
          this.links.push(link);
        }
      }
    }
    this.active = []; // input neurons that are non-zero for the current review
  }

  get inputLayer() { return this.layers[0]; }
  get outputLayer() { return this.layers[this.layers.length - 1]; }

  // input = { idx: [neuron indexes], val: [their values] }  (only the non-zero inputs)
  forward(input) {
    for (const n of this.active) n.output = 0;
    this.active = [];
    const inputs = this.inputLayer.neurons;
    for (let k = 0; k < input.idx.length; k++) {
      const n = inputs[input.idx[k]];
      n.output = input.val[k];
      this.active.push(n);
    }

    // First layer after the input: start each neuron at its bias, then let
    // only the active input neurons push signal through their links.
    const first = this.layers[1];
    for (const n of first.neurons) n.sum = n.bias;
    for (const a of this.active) {
      for (const link of a.outgoing) link.to.sum += a.output * link.weight;
    }
    for (const n of first.neurons) n.output = sigmoid(n.sum);

    // Remaining layers are small, so they fire densely.
    for (let l = 2; l < this.layers.length; l++) {
      for (const n of this.layers[l].neurons) n.fire();
    }
    return this.outputLayer.neurons[0].output;
  }

  // Compute every neuron's delta (blame). Weights are NOT changed here.
  backward(target) {
    const out = this.outputLayer.neurons[0];
    // sigmoid output + binary cross-entropy loss => delta = prediction - target
    out.delta = out.output - target;

    for (let l = this.layers.length - 2; l >= 1; l--) {
      for (const n of this.layers[l].neurons) {
        let blame = 0;
        for (const link of n.outgoing) blame += link.weight * link.to.delta;
        n.delta = blame * n.output * (1 - n.output); // × slope of sigmoid
      }
    }
  }

  // Nudge weights and biases using the deltas from backward().
  // new_weight = old_weight - learningRate × delta(receiving neuron) × output(sending neuron)
  update(learningRate) {
    for (let l = 2; l < this.layers.length; l++) {
      for (const n of this.layers[l].neurons) {
        for (const link of n.incoming) link.weight -= learningRate * n.delta * link.from.output;
        n.bias -= learningRate * n.delta;
      }
    }
    // Links out of the input layer: only the active ones can change.
    for (const a of this.active) {
      for (const link of a.outgoing) link.weight -= learningRate * link.to.delta * a.output;
    }
    for (const n of this.layers[1].neurons) n.bias -= learningRate * n.delta;
  }

  // One training step. Returns the prediction made BEFORE the update.
  train(input, target, learningRate) {
    const prediction = this.forward(input);
    this.backward(target);
    this.update(learningRate);
    return prediction;
  }

  getWeights() { return Float64Array.from(this.links, k => k.weight); }
  getBiases() { return this.layers.slice(1).map(L => L.neurons.map(n => n.bias)); }

  toCheckpoint() {
    return { sizes: this.sizes, weights: Array.from(this.getWeights()), biases: this.getBiases() };
  }

  // Per-neuron view of the whole model: every non-input neuron with its bias
  // and the weight on each incoming link. Weights into the first hidden layer
  // are keyed by the word they come from; later layers list one weight per
  // sending neuron, in order.
  toNeuronJSON(inputLabels) {
    return this.layers.slice(1).map((L, li) => ({
      layer: L.name,
      inputs_from: li === 0 ? 'words (key = word)' : `${this.layers[li].name} neurons 0..${this.layers[li].neurons.length - 1} (in order)`,
      neurons: L.neurons.map((n, i) => ({
        neuron: i,
        bias: n.bias,
        weights: li === 0
          ? Object.fromEntries(n.incoming.map((k, j) => [inputLabels ? inputLabels[j] : j, k.weight]))
          : n.incoming.map(k => k.weight),
      })),
    }));
  }

  static fromCheckpoint(cp, inputLabels = null) {
    if (cp.layers) {
      // per-neuron format written by toNeuronJSON()
      const net = new Network(cp.sizes, { inputLabels });
      cp.layers.forEach((L, li) => L.neurons.forEach((nd, ni) => {
        const n = net.layers[li + 1].neurons[ni];
        n.bias = nd.bias;
        const ws = Array.isArray(nd.weights) ? nd.weights : cp.vocabulary.map(w => nd.weights[w]);
        n.incoming.forEach((link, j) => { link.weight = ws[j]; });
      }));
      return net;
    }
    const net = new Network(cp.sizes, { inputLabels });
    net.links.forEach((link, i) => { link.weight = cp.weights[i]; });
    cp.biases.forEach((arr, li) => arr.forEach((b, ni) => { net.layers[li + 1].neurons[ni].bias = b; }));
    return net;
  }
}

module.exports = { Network, Neuron, Link, Layer, sigmoid, mulberry32 };
