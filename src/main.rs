mod node {
    use std::cell::{Ref, RefCell, RefMut};
    use std::fmt::Debug;
    use std::ops::{Add, Mul};
    use std::rc::Rc;

    struct Input {
        value: f64,
    }

    impl Input {
        fn new(value: f64) -> Self {
            Self { value }
        }

        fn set_value(&mut self, value: f64) {
            self.value = value;
        }
    }

    enum Op {
        Input(Input),
        Plus(Node, Node),
        Times(Node, Node),
        Tanh(Node),
    }

    struct NodeInner {
        op: Op,
        value: Option<f64>,
        grad: Option<f64>,
    }

    impl Debug for NodeInner {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self)
        }
    }

    impl NodeInner {
        #[cfg(test)]
        fn set_value(&mut self, value: f64) {
            self.value = Some(value);
        }

        #[cfg(test)]
        fn set_grad(&mut self, grad: f64) {
            self.grad = Some(grad);
        }

        fn eval(&mut self) -> f64 {
            match &mut self.op {
                Op::Input(input) => {
                    self.value = Some(input.value);
                }
                Op::Plus(x, y) => {
                    self.value = Some(x.eval() + y.eval());
                }
                Op::Times(x, y) => {
                    self.value = Some(x.eval() * y.eval());
                }
                Op::Tanh(x) => {
                    let x = x.eval();
                    // Let's write this out
                    let num = f64::exp(2.0 * x) - 1.0;
                    let den = f64::exp(2.0 * x) + 1.0;
                    let value = num / den;
                    self.value = Some(value);
                }
            }
            self.value.expect("Value is not set when evaling?")
        }

        fn reset_grads(&mut self) {
            self.grad = None;
            match &mut self.op {
                Op::Input(_) => (),
                Op::Plus(x, y) => {
                    x.reset_grads();
                    y.reset_grads();
                }
                Op::Times(x, y) => {
                    x.reset_grads();
                    y.reset_grads();
                }
                Op::Tanh(x) => {
                    x.reset_grads();
                }
            }
        }

        fn compute_grads(&mut self, input_grad: f64) {
            let current_grad = self.grad.unwrap_or(0.0);
            self.grad = Some(current_grad + input_grad);

            match &mut self.op {
                Op::Input(_) => (),
                Op::Plus(x, y) => {
                    x.compute_grads(input_grad);
                    y.compute_grads(input_grad);
                }
                Op::Times(x, y) => {
                    let x_value = x.value().expect("Value is not set when computing grads?");
                    let y_value = y.value().expect("Value is not set when computing grads?");
                    x.compute_grads(input_grad * y_value);
                    y.compute_grads(input_grad * x_value);
                }
                Op::Tanh(x) => {
                    let value = x.value().expect("Value is not set when computing grads?");
                    let local_grad = 1.0 - (value * value);
                    let output_grad = local_grad * input_grad;
                    x.compute_grads(output_grad);
                }
            }
        }
    }

    pub struct Node {
        // Possibly replace this with unsafe raw pointer for speed?
        inner: Rc<RefCell<NodeInner>>,
    }

    impl Debug for Node {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.inner.borrow())
        }
    }

    impl Node {
        fn new(inner: NodeInner) -> Self {
            Self {
                inner: Rc::new(RefCell::new(inner)),
            }
        }

        fn get(&self) -> Ref<NodeInner> {
            self.inner.borrow()
        }

        fn get_mut(&mut self) -> RefMut<NodeInner> {
            self.inner.borrow_mut()
        }

        pub fn eval(&mut self) -> f64 {
            self.get_mut().eval()
        }

        pub fn reset_grads(&mut self) {
            self.get_mut().reset_grads();
        }

        pub fn compute_grads(&mut self, input_grad: f64) {
            self.get_mut().compute_grads(input_grad);
        }

        pub fn value(&self) -> Option<f64> {
            self.get().value
        }

        pub fn input(input: Input) -> Self {
            Self::new(NodeInner {
                op: Op::Input(input),
                value: None,
                grad: None,
            })
        }

        fn plus(x: &Self, y: &Self) -> Self {
            Self::new(NodeInner {
                op: Op::Plus(x.clone(), y.clone()),
                value: None,
                grad: None,
            })
        }

        fn times(x: &Self, y: &Self) -> Self {
            Self::new(NodeInner {
                op: Op::Times(x.clone(), y.clone()),
                value: None,
                grad: None,
            })
        }
    }

    impl Clone for Node {
        fn clone(&self) -> Self {
            Self {
                inner: self.inner.clone(),
            }
        }
    }

    impl Add for &Node {
        type Output = Node;

        fn add(self, other: Self) -> Node {
            Node::plus(self, other)
        }
    }

    impl Mul for &Node {
        type Output = Node;

        fn mul(self, other: Self) -> Node {
            Node::times(self, other)
        }
    }
}

fn main() {
    println!("Aiis, edhor!");
}
