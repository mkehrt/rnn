use crate::node::{new_input, new_weight, Weight, Input, Node};

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

    fn instance(&self, inputs: &Vec<Node>) -> Node {
        assert_eq!(inputs.len(), self.size);

        let input_nodes = inputs.clone();

        let weights = self.weights.clone();
        let weight_nodes: Vec<Node> = weights.iter().map(|weight| Node::weight(weight.clone())).collect();

        let factors = input_nodes.iter().zip(weight_nodes.iter());
        let product = factors.map(|(input, weight)| input * weight);
        let bias_node = Node::weight(self.bias.clone());
        let root: Node = product.fold(bias_node, |acc, node| &acc + &node);

        root
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

    fn instance(&self, inputs: &Vec<Node>) -> Vec<Node> {
        let mut nodes = Vec::new();
        for neuron in &self.neurons {
            nodes.push(neuron.instance(inputs));
        }
        nodes
    }
}

struct Perceptron {
    input_size: usize,
    layers: Vec<Layer>,
}

impl Perceptron {
    fn new(input_size: usize, layer_sizes: Vec<usize>) -> Self {
        let mut layers = Vec::new();

        let mut sizes = vec![input_size];
        sizes.extend(layer_sizes);
        let size_pairs = sizes.windows(2);
        for size_pair in size_pairs {
            let input_size = size_pair[0];
            let output_size = size_pair[1];
            layers.push(Layer::new(input_size, /* neuron size */ output_size));
        }

        Self { input_size, layers }
    }

    // Returns output nodes.
    fn instance(&self, inputs: &Vec<Node>) -> Vec<Node> {
        let mut nodes = inputs.clone();
        for layer in &self.layers {
            nodes = layer.instance(&nodes);
        }
        nodes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptron() {
        // 1:51:08
        let perceptron = Perceptron::new(3, vec![4, 4, 1]);

        let inputs_to_outputs = vec![
            (vec![2.0, 3.0, -1.0], 1.0),
            (vec![3.0, -5.0, 0.5], -1.0),
            (vec![0.5, 3.0, 1.0], -1.0),
            (vec![1.0, 1.0, -3.0], 1.0),
        ];

    }
}