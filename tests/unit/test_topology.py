from firm_ce.system.scalar.topology import Node


def test_node_instantiation():
    node = Node(static_instance=True, idx=1234, order=5678, name="Nodey_McNodeyface")

    assert isinstance(node, Node)
