import numpy as np
class Node(object):
    def __init__(self,name,value=0,node_type=0):
        self.name=name
        self.node_type=node_type
        self.value = value
        if not self.node_type:
            self.value=[]
    def add_node(self,node):
        assert not self.node_type,'叶节点不允许插入'
        self.value.append(node)

    def __getitem__(self, item):
        return self.value[item]





node=   Node('A',value=1)
node1= Node('A1',value=2,node_type=1)
node2=Node('A2',value=1,node_type=1)

node.add_node(node1)
node.add_node(node2)
print([i.value for i in node])

