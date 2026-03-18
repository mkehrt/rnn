mod node {
    use std::rc::Rc;
    use std::ops::Add;
    enum Op {
        Input,
        Plus(Node, Node),
        Times(Node, Node),
    }

    struct NodeInner {
        op: Op,
        data: Option<f64>,
        grad: Option<f64>,
    }

    pub struct Node(Rc<NodeInner>);

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
}

fn main() {
    println!("Aiis, edhor!");
}
