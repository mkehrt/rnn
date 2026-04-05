use float_cmp::approx_eq;
use std::cell::{Ref, RefCell, RefMut};
use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};
use std::rc::Rc;

fn f64_eq(a: f64, b: f64) -> bool {
    approx_eq!(f64, a, b, epsilon = 1e-6)
}

fn f64_option_eq(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => f64_eq(a, b),
        (None, None) => true,
        _ => false,
    }
}

pub struct Input {
    inner: Rc<RefCell<f64>>,
}

impl Input {
    fn new(value: f64) -> Self {
        Self {
            inner: Rc::new(RefCell::new(value)),
        }
    }

    fn value(&self) -> f64 {
        *self.inner.borrow()
    }

    fn set_value(&self, value: f64) {
        *self.inner.borrow_mut() = value;
    }
}

impl Clone for Input {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl Debug for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Input({:?})", self.value())
    }
}

impl PartialEq for Input {
    fn eq(&self, other: &Self) -> bool {
        f64_eq(self.value(), other.value())
    }
}

#[derive(Debug, PartialEq)]
enum Op {
    Input(Input),
    Plus(Node, Node),
    Minus(Node, Node),
    Times(Node, Node),
    Tanh(Node),
}

struct NodeInner {
    op: Op,
    value: Option<f64>,
    grad: Option<f64>,
}

impl PartialEq for NodeInner {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op
            && f64_option_eq(self.value, other.value)
            && f64_option_eq(self.grad, other.grad)
    }
}

impl Debug for NodeInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
         f.debug_struct("Node")
            .field("op", &self.op)
            .field("value", &self.value)
            .field("grad", &self.grad)
            .finish()
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
                self.value = Some(input.value());
            }
            Op::Plus(x, y) => {
                self.value = Some(x.eval() + y.eval());
            }
            Op::Minus(x, y) => {
                self.value = Some(x.eval() - y.eval());
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
            Op::Minus(x, y) => {
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
            Op::Minus(x, y) => {
                x.compute_grads(input_grad);
                y.compute_grads(-input_grad);
            }
            Op::Times(x, y) => {
                let x_value = x.value().expect("Value is not set when computing grads?");
                let y_value = y.value().expect("Value is not set when computing grads?");
                x.compute_grads(input_grad * y_value);
                y.compute_grads(input_grad * x_value);
            }
            Op::Tanh(x) => {
                let value = self.value.expect("Value is not set when computing grads?");
                let local_grad = 1.0 - (value * value);
                let output_grad = local_grad * input_grad;
                x.compute_grads(output_grad);
            }
        }
    }
}

#[derive(PartialEq)]
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
    fn new(op: Op) -> Self {
        Self {
            inner: Rc::new(RefCell::new(NodeInner {
                op,
                value: None,
                grad: None,
            })),
        }
    }

    // Weird type here to allow
    //   let x = (y + z).set_value(1.0).set_grad(2.0);
    #[cfg(test)]
    fn set_value(self, value: f64) -> Self {
        self.get_mut().set_value(value);
        self
    }

    // Weird type here to allow
    //   let x = (y + z).set_value(1.0).set_grad(2.0);
    #[cfg(test)]
    fn set_grad(self, grad: f64) -> Self {
        self.get_mut().set_grad(grad);
        self
    }

    fn get(&self) -> Ref<NodeInner> {
        self.inner.borrow()
    }

    fn get_mut(&self) -> RefMut<NodeInner> {
        self.inner.borrow_mut()
    }

    pub fn eval(&self) -> f64 {
        self.get_mut().eval()
    }

    pub fn reset_grads(&self) {
        self.get_mut().reset_grads();
    }

    pub fn compute_grads(&self, input_grad: f64) {
        self.get_mut().compute_grads(input_grad);
    }

    pub fn value(&self) -> Option<f64> {
        self.get().value
    }

    pub fn input(input: Input) -> Self {
        Self::new(Op::Input(input))
    }

    fn plus(x: &Self, y: &Self) -> Self {
        Self::new(Op::Plus(x.clone(), y.clone()))
    }

    fn times(x: &Self, y: &Self) -> Self {
        Self::new(Op::Times(x.clone(), y.clone()))
    }

    fn minus(x: &Self, y: &Self) -> Self {
        Self::new(Op::Minus(x.clone(), y.clone()))
    }

    pub fn tanh(&self) -> Self {
        Self::new(Op::Tanh(self.clone()))
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

impl Sub for &Node {
    type Output = Node;

    fn sub(self, other: Self) -> Node {
        Node::minus(self, other)
    }
}

impl Mul for &Node {
    type Output = Node;

    fn mul(self, other: Self) -> Node {
        Node::times(self, other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input() {
        let input = Input::new(1.0);
        let i = Node::input(input);

        let input_expected = Input::new(1.0);
        let i_expected = Node::input(input_expected).set_value(1.0).set_grad(1.0);

        i.eval();
        i.reset_grads();
        i.compute_grads(1.0);

        assert_eq!(i, i_expected);
    }

    #[test]
    fn test_first_example() {
        // https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
        // Around 51:45.
        let a_input = Input::new(2.0);
        let b_input = Input::new(-3.0);
        let c_input = Input::new(10.0);
        let f_input = Input::new(-2.0);

        let a = Node::input(a_input);
        let b = Node::input(b_input);
        let c = Node::input(c_input);
        let f = Node::input(f_input);

        let e = &a * &b;
        let d = &e + &c;
        #[allow(non_snake_case)]
        let L = &d * &f;

        let a_input_expected = Input::new(2.0);
        let b_input_expected = Input::new(-3.0);
        let c_input_expected = Input::new(10.0);
        let f_input_expected = Input::new(-2.0);

        let a_expected = Node::input(a_input_expected).set_value(2.0).set_grad(6.0);
        let b_expected = Node::input(b_input_expected).set_value(-3.0).set_grad(-4.0);
        let c_expected = Node::input(c_input_expected).set_value(10.0).set_grad(-2.0);
        let f_expected = Node::input(f_input_expected).set_value(-2.0).set_grad(4.0);

        let e_expected = (&a_expected * &b_expected).set_value(-6.0).set_grad(-2.0);
        let d_expected = (&e_expected + &c_expected).set_value(4.0).set_grad(-2.0);
        #[allow(non_snake_case)]
        let L_expected = (&d_expected * &f_expected).set_value(-8.0).set_grad(1.0);

        L.eval();
        L.reset_grads();
        L.compute_grads(1.0);

        assert_eq!(L, L_expected);
    }

    #[test]
    fn test_tanh() {
        // https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
        // Around 1:04:40.

        // atanh(sqrt(2)/2)
        let input = 0.8813735870195432;
        // sqrt(2)/2
        let output = 0.7071067811865476;

        let x_input = Input::new(input);
        let x = Node::input(x_input);

        let y = x.tanh();

        let x_input_expected = Input::new(input);
        let x_expected = Node::input(x_input_expected).set_value(input).set_grad(0.5);
        let y_expected = x_expected.tanh().set_value(output).set_grad(1.0);

        y.eval();
        y.reset_grads();
        y.compute_grads(1.0);
        assert_eq!(y, y_expected);
    }

    #[test]
    fn test_dag() {
        // https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
        // Around 1:26:17.

        let a_input = Input::new(-2.0);
        let b_input = Input::new(3.0);

        let a = Node::input(a_input);
        let b = Node::input(b_input);

        let c = &a + &b;
        let d: Node = &a * &b;
        
        // e???
        let f = &c * &d;

        let a_input_expected = Input::new(-2.0);
        let b_input_expected = Input::new(3.0);

        let a_expected = Node::input(a_input_expected).set_value(-2.0).set_grad(-3.0);
        let b_expected = Node::input(b_input_expected).set_value(3.0).set_grad(-8.0);

        let c_expected = (&a_expected + &b_expected).set_value(1.0).set_grad(-6.0);
        let d_expected = (&a_expected * &b_expected).set_value(-6.0).set_grad(1.0);
        
        // e???
        let f_expected = (&c_expected * &d_expected).set_value(-6.0).set_grad(1.0);
        
        f.eval();
        f.reset_grads();
        f.compute_grads(1.0);
        assert_eq!(f, f_expected);
    }
}
