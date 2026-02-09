import importlib.util
import os
import unittest
from unittest.mock import patch


def load_module():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    module_path = os.path.join(repo_root, "tools", "ascend_numa_bind.py")
    spec = importlib.util.spec_from_file_location("ascend_numa_bind", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAscendNumaBind(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mod = load_module()

    def test_compress_cpus(self):
        self.assertEqual(self.mod.compress_cpus([]), "")
        self.assertEqual(self.mod.compress_cpus([1, 2, 3]), "1-3")
        self.assertEqual(self.mod.compress_cpus([0, 2, 3, 4, 6]), "0,2-4,6")
        self.assertEqual(self.mod.compress_cpus([3, 1, 2, 2, 5]), "1-3,5")

    def test_chunk_list(self):
        self.assertEqual(self.mod.chunk_list([1, 2, 3, 4], 2), [[1, 2], [3, 4]])
        self.assertEqual(self.mod.chunk_list([1, 2, 3], 2), [[1, 2], [3]])
        self.assertEqual(self.mod.chunk_list([1, 2], 0), [[1, 2]])

    @patch("ascend_numa_bind.run_cmd")
    def test_parse_lscpu(self, mock_run_cmd):
        mock_run_cmd.return_value = ("0 0 0\n1 0 0\n2 1 0\n3 1 0\n", 0)
        socket_map = self.mod.parse_lscpu()
        expected = {0: {0: [0, 1], 1: [2, 3]}}
        self.assertEqual(socket_map, expected)

    def test_build_group_plans(self):
        socket_map = {0: {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}}
        plans = self.mod.build_group_plans(socket_map, groups_per_socket=2)
        self.assertEqual(len(plans), 2)
        self.assertEqual(plans[0].primary_node, 0)
        self.assertEqual(plans[0].secondary_node, 1)
        self.assertEqual(plans[0].primary_cpus, [0, 1])
        self.assertEqual(plans[1].primary_cpus, [2, 3])
        self.assertEqual(plans[0].secondary_cpus, [4, 5, 6, 7])

    def test_assign_processes_to_groups(self):
        group_a = self.mod.GroupPlan(
            name="g0",
            socket=0,
            group_index=0,
            primary_node=0,
            secondary_node=1,
            primary_cpus=[0, 1],
            secondary_cpus=[2, 3],
            mem_nodes=[0, 1],
        )
        group_b = self.mod.GroupPlan(
            name="g1",
            socket=0,
            group_index=1,
            primary_node=0,
            secondary_node=1,
            primary_cpus=[4, 5],
            secondary_cpus=[2, 3],
            mem_nodes=[0, 1],
        )
        procs = [
            self.mod.ProcessInfo(pid=300, comm="a", args="a"),
            self.mod.ProcessInfo(pid=100, comm="b", args="b"),
            self.mod.ProcessInfo(pid=200, comm="c", args="c"),
        ]
        assigned = self.mod.assign_processes_to_groups(procs, [group_a, group_b])
        self.assertEqual(assigned[100].name, "g0")
        self.assertEqual(assigned[200].name, "g1")
        self.assertEqual(assigned[300].name, "g0")


if __name__ == "__main__":
    unittest.main()
