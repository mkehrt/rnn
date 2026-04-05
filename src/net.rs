use crate::node::{new_input, new_weight, Weight, Input, Node};

trait Module {
    fn instance(&self) -> impl ModuleInstance;
    fn weights(&self) -> Vec<Weight>;
}

trait ModuleInstance {
    type Output;

    fn set_inputs(&self, inputs: &[f64]);
    fn eval(&self) -> Self::Output;
    fn reset_grads(&self);
    fn compute_grads(&self, input_grad: Self::Output);
}

struct Neuron {
    size: usize,
    weights: Vec<Weight>,
    bias: Weight,
}

impl Neuron {
    fn new(size: usize) -> Self {
        Self {
            size,
            weights: vec![new_weight(0.0); size],
            bias: new_weight(0.0),
        }
    }
}

impl Module for Neuron {
    fn instance(&self) -> NeuronInstance {
        let inputs = vec![new_input(0.0); self.size];
        let returned_inputs = inputs.clone();
        let input_nodes: Vec<Node> = returned_inputs.iter().map(|input| Node::input(input.clone())).collect();

        let weights = self.weights.clone();
        let weight_nodes: Vec<Node> = weights.iter().map(|weight| Node::weight(weight.clone())).collect();

        let factors = input_nodes.iter().zip(weight_nodes.iter());
        let product = factors.map(|(input, weight)| input * weight);
        let bias_node = Node::weight(self.bias.clone());
        let root: Node = product.fold(bias_node, |acc, node| &acc + &node);

        NeuronInstance {
            size: self.size,
            inputs: returned_inputs,
            root,
        }
    }

    fn weights(&self) -> Vec<Weight> {
        let mut weights = self.weights.clone();
        weights.push(self.bias.clone());
        weights
    }
}

struct NeuronInstance {
    size: usize,
    inputs: Vec<Input>,
    root: Node,
}

impl ModuleInstance for NeuronInstance {
    type Output = f64;

    fn set_inputs(&self, inputs: &[f64]) {
        assert_eq!(inputs.len(), self.size);
        for (input, value) in self.inputs.iter().zip(inputs.iter()) {
            input.set_value(*value);
        }
    }
    fn eval(&self) -> f64 {
        self.root.eval()
    }
    fn reset_grads(&self) {
        self.root.reset_grads()
    }
    fn compute_grads(&self, input_grad: f64) {
        self.root.compute_grads(input_grad)
    }
}


struct Layer {
    neurons: Vec<Neuron>,
    size: usize,
}

impl Layer {
    fn new(size: usize, neuron_size: usize) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..size {
            neurons.push(Neuron::new(neuron_size));
        }

        Self { neurons, size }
    }
}

impl Module for Layer {
    fn instance(&self) -> LayerInstance {
        let neurons = self.neurons.iter().map(|neuron| neuron.instance()).collect();
        LayerInstance {
            neurons,
        }
    }
    
    fn weights(&self) -> Vec<Weight> {
        let mut weights = vec![];
        for neuron in &self.neurons {
            weights.extend(neuron.weights());
        }
        weights
    }
}

struct LayerInstance {
    neurons: Vec<NeuronInstance>,
}

impl ModuleInstance for LayerInstance {
    type Output = Vec<f64>;

    fn set_inputs(&self, inputs: &[f64]) {
        for neuron in &self.neurons {
            neuron.set_inputs(inputs);
        }
    }

    fn eval(&self) -> Vec<f64> {
        self.neurons.iter().map(|neuron| neuron.eval()).collect()
    }

    fn reset_grads(&self) {
        for neuron in &self.neurons {
            neuron.reset_grads();
        }
    }

    fn compute_grads(&self, input_grads: Vec<f64>) {
        for neuron in &self.neurons {
            for input_grad in input_grads.iter() {
                neuron.compute_grads(*input_grad);
            }
        }
    }
}

struct Perceptron {
    input_size: usize,
    layers: Vec<Layer>,
}

impl Perceptron {
    fn new(input_size: usize, layer_sizes: Vec<usize>) -> Self {
        let mut layers = Vec::new();

        let sizes = vec![input_size] + layer_sizes;
        let size_pairs = sizes.windows(2);
        for (input_size, output_size) in size_pairs {
            layers.push(Layer::new(*input_size, /* neuron size */*output_size));
        }

        Self { input_size, layers }
    }
}

impl Module for Perceptron {
    fn instance(&self) -> PerceptronInstance {
        let mut layers = Vec::new();
        for layer in &self.layers {
            layers.push(layer.instance());
        }
        PerceptronInstance { input_size: self.input_size, layers }
    }

    fn weights(&self) -> Vec<Weight> {
        let mut weights = vec![];
        for layer in &self.layers {
            weights.extend(layer.weights());
        }
        weights
    }
}

struct PerceptronInstance {
    input: Option<Vec<f64>>,
    layers: Vec<LayerInstance>,
}

impl ModuleInstance for PerceptronInstance {
    type Output = Vec<f64>;

    fn set_inputs(&self, inputs: &[f64]) {
        self.input = Some(inputs.to_vec());
    }

    fn eval(&self) -> Vec<f64> {
        assert!(self.input.is_some());
        let mut next_inputs = self.input.unwrap();

        for layer in &self.layers {
            layer.set_inputs(&next_inputs);
            next_inputs = layer.eval();
        }
        next_inputs
    }

    fn reset_grads(&self) {
        for layer in &self.layers {
            layer.reset_grads();
        }
    }

    fn compute_grads(&self, input_grads: Vec<f64>) {
        let mut next_grads = 1.0;

        for layer in &self.layers.iter().rev() {
            layer.compute_grads(&input_grads);
            next_input = layer.eval();
        }
    }
}