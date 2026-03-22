mod node {
    use std::fmt::Debug;
    use std::ops::{Add, Mul};
    use std::cell::{RefCell, Ref, RefMut};
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

    impl Debug for NodeInner {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:?}", self)
        }
    }

    impl NodeInner {
        fn eval(&mut self) -> f64 {
            match &mut self.op {
                Op::Input(input) => {
                    self.data = Some(input.value);
                }
                Op::Plus(a, b) => {
                    self.data = Some(a.eval() + b.eval());
                }
                Op::Times(a, b) => {
                    self.data = Some(a.eval() * b.eval());
                }
            }
            self.data.expect("Data is not set when evaling?")
        }

        fn reset_grads(&mut self) {
            self.grad = None;
            match &mut self.op {
                Op::Input(_) => (),
                Op::Plus(a, b) => {
                    a.reset_grads();
                    b.reset_grads();
                }
                Op::Times(a, b) => {
                    a.reset_grads();
                    b.reset_grads();
                }
            }
        }

        fn compute_grads(&mut self, input_grad: f64) {
            let current_grad = self.grad.unwrap_or(0.0);
            self.grad = Some(current_grad + input_grad);

            match &mut self.op {
                Op::Input(_) => self.grad = Some(input_grad),
                Op::Plus(a, b) => {
                    a.compute_grads(input_grad);
                    b.compute_grads(input_grad);
                }
                Op::Times(a, b) => {
                    let a_data = a.data().expect("Data is not set when computing grads?");
                    let b_data = b.data().expect("Data is not set when computing grads?");
                    a.compute_grads(input_grad * b_data);
                    b.compute_grads(input_grad * a_data);
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
            Self { inner: Rc::new(RefCell::new(inner)) }
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

        pub fn data(&self) -> Option<f64> {
            self.get().data
        }

        pub fn input(input: Input) -> Self {
            Self::new(NodeInner {
                op: Op::Input(input),
                data: None,
                grad: None,
            })
        }

        fn plus(a: &Self, b: &Self) -> Self {
            Self::new(NodeInner {
                op: Op::Plus(a.clone(), b.clone()),
                data: None,
                grad: None,
            })
        }

        fn times(a: &Self, b: &Self) -> Self {
            Self::new(NodeInner {
                op: Op::Times(a.clone(), b.clone()),
                data: None,
                grad: None,
            })
        }
    }

    impl Clone for Node {
        fn clone(&self) -> Self {
            Self { inner: self.inner.clone() }
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
