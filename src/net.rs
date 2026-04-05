use crate::node::{new_input, new_weight, Weight, Input, Node};

struct Neuron {
    size: usize,
    weights: Vec<Weight>,
    bias: Weight,
}

struct Instance {
    inputs: Vec<Input>,
    root: Node,
}

impl Neuron {
    fn new(size: usize) -> Self {
        Self {
            size,
            weights: vec![new_weight(0.0); size],
            bias: new_weight(0.0),
        }
    }

    fn all_weights(&self) -> Vec<Weight> {
        let mut weights = self.weights.clone();
        weights.push(self.bias.clone());
        weights
    }

    fn get_instance(&self) -> Instance {
        let inputs = vec![new_input(0.0); self.size];
        let returned_inputs = inputs.clone();
        let input_nodes: Vec<Node> = returned_inputs.iter().map(|input| Node::input(input.clone())).collect();

        let weights = self.weights.clone();
        let weight_nodes: Vec<Node> = weights.iter().map(|weight| Node::weight(weight.clone())).collect();

        let factors = input_nodes.iter().zip(weight_nodes.iter());
        let product = factors.map(|(input, weight)| input * weight);
        let bias_node = Node::weight(self.bias.clone());
        let root: Node = product.fold(bias_node, |acc, node| &acc + &node);

        Instance {
            inputs: returned_inputs,
            root,
        }
    }
}