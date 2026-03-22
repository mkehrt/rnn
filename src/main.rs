mod node {
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
    }

    struct NodeInner {
        op: Op,
        data: Option<f64>,
        grad: Option<f64>,
    }

    impl NodeInner {
        fn eval(&self) -> f64 {
            match &self.op {
                Op::Input(input) => {
                    self.data = Some(input.value);
                }
                Op::Plus(a, b) => {
                    self.data = a.eval() + b.eval();
                }
                Op::Times(a, b) => {
                    self.data = a.eval() * b.eval();
                }
            }
            self.data.expect("Data is not set when evalling?")
        }

        fn compute_grads(&self, input_grad: f64) {
            let current_grad = self.grad.unwrap_or(0.0);
            self.grad = Some(current_grad + input_grad);

            match &self.op {
                Op::Input(_) => self.grad = Some(input_grad),
                Op::Plus(a, b) => {
                    a.compute_grads(input_grad);
                    b.compute_grads(input_grad);
                }
                Op::Times(a, b) => {
                    let a_data = a.data.expect("Data is not set when computing grads?");
                    let b_data = b.data.expect("Data is not set when computing grads?");
                    a.compute_grads(input_grad * b_data);
                    b.compute_grads(input_grad * a_data);
                }
            }
        }
    }

    pub struct Node {
        inner: Rc<NodeInner>,
    }

    impl Debug for Node {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self.inner)
        }
    }

    impl Node {
        pub fn input() -> Self {
            Self(Rc::new(NodeInner {
                op: Op::Input,
                data: None,
                grad: None,
            }))
        }

        fn plus(a: &Self, b: &Self) -> Self {
            Self(Rc::new(NodeInner {
                op: Op::Plus(a.clone(), b.clone()),
                data: None,
                grad: None,
            }))
        }

        fn times(a: &Self, b: &Self) -> Self {
            Self(Rc::new(NodeInner {
                op: Op::Times(a.clone(), b.clone()),
                data: None,
                grad: None,
            }))
        }
    }

    impl Clone for Node {
        fn clone(&self) -> Self {
            Self(self.0.clone())
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
