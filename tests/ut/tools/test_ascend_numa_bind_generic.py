import importlib.util
import os
import unittest
from unittest.mock import patch


def load_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    module_path = os.path.join(repo_root, "tools", "ascend_numa_bind_generic.py")
    spec = importlib.util.spec_from_file_location("ascend_numa_bind_generic", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAscendNumaBindGeneric(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    @patch("ascend_numa_bind_generic.parse_node_distances")
    def test_pick_secondary_node(self, mock_dist):
        mock_dist.return_value = [10, 20, 15]
        nodes = [0, 1, 2]
        self.assertEqual(self.mod.pick_secondary_node(0, nodes), 2)

    def test_build_group_plans_single_node(self):
        socket_map = {0: {0: [0, 1, 2, 3]}}
        plans = self.mod.build_group_plans(socket_map, groups_per_socket=2)
        self.assertEqual(len(plans), 2)
        for plan in plans:
            self.assertEqual(plan.primary_node, 0)
            self.assertEqual(plan.secondary_node, 0)
            self.assertEqual(plan.mem_nodes, [0])

    @patch("ascend_numa_bind_generic.parse_node_distances")
    def test_build_group_plans_multi_node(self, mock_dist):
        mock_dist.return_value = [10, 20]
        socket_map = {0: {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}}
        plans = self.mod.build_group_plans(socket_map, groups_per_socket=2)
        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0].primary_node, 0)
        self.assertEqual(plans[0].secondary_node, 1)
        self.assertEqual(plans[0].secondary_cpus, [4, 5, 6, 7])


if __name__ == "__main__":
    unittest.main()
