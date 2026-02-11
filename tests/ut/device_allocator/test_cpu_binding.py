import unittest
from unittest.mock import patch

from vllm_ascend.cpu_binding import CpuAlloc, DeviceInfo, get_current_npu_id
from vllm_ascend.utils import AscendDeviceType


class TestDeviceInfo(unittest.TestCase):

    @patch('vllm_ascend.cpu_binding.execute_command')
    def setUp(self, mock_execute_command):
        mock_execute_command.side_effect = [
            ("NPU ID  Chip ID  Chip Logic ID  Chip Name\n0 0 0 Ascend\n0 1 - Mcu\n1 0 1 Ascend",
             0),
            ("| NPU Chip | Process id |\n| 0 0 | 1234 | vllm | 56000 |\n| 1 0 | 1235 | vllm | 56000 |",
             0), ("", 0)
        ]
        self.device_info = DeviceInfo()

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_get_npu_map_info(self, mock_execute_command):
        execute_result_list = [
            ("NPU ID  Chip ID  Chip Logic ID  Chip Phy-ID Chip Name\n0 0 0 0 Ascend\n0 1 1 1 Ascend\n0 2 - - Mcu",
             0),
            ("NPU ID  Chip ID  Chip Logic ID  Chip Name\n8 0 0 Ascend\n8 1 - Mcu\n9 0 1 Ascend",
             0),
        ]
        result_list = [{
            '0': {
                '0': '0',
                '1': '1'
            }
        }, {
            '8': {
                '0': '0'
            },
            '9': {
                '0': '1'
            }
        }]
        for result in execute_result_list:
            mock_execute_command.return_value = result
            npu_map_info = self.device_info.get_npu_map_info()
            expected = result_list.pop(0)
            self.assertEqual(npu_map_info, expected)

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_get_running_npus(self, mock_execute_command):
        mock_execute_command.side_effect = [
            ("| NPU Chip | Process id |\n| 0 0 | 1236 | vllm | 56000 |", 0),
            ("", 0),
            ("| NPU Chip | Process id |\n| 1 0 | 1236 | vllm | 56000 |", 0)
        ]
        running_npus = self.device_info.get_running_npus()
        self.assertEqual(running_npus, [0, 1])
        running_npus = self.device_info.get_running_npus()
        self.assertEqual(running_npus, [0, 1])
        running_npus = self.device_info.get_running_npus()
        self.assertEqual(running_npus, [0, 1])

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_get_running_npus_fallback_visible(self, mock_execute_command):
        mock_execute_command.return_value = ("| NPU Chip | Process id |", 0)
        running_npus = self.device_info.get_running_npus()
        self.assertEqual(running_npus, [0, 1])

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_parse_topo_affinity(self, mock_execute_command):
        mock_execute_command.return_value = (
            "NPU0 X HCCS HCCS HCCS HCCS HCCS HCCS HCCS 0-3", 0)
        affinity = self.device_info.parse_topo_affinity()
        expected = {0: [0, 1, 2, 3]}
        self.assertEqual(affinity, expected)

    def test_expand_cpu_list(self):
        result = self.device_info.expand_cpu_list("0-2, 4, 6-8")
        self.assertEqual(result, [0, 1, 2, 4, 6, 7, 8])

    @patch('vllm_ascend.cpu_binding.ASCEND_RT_VISIBLE_DEVICES', '3')
    def test_get_current_npu_id_single_visible(self):
        self.assertEqual(get_current_npu_id(0), 3)

    @patch('vllm_ascend.cpu_binding.ASCEND_RT_VISIBLE_DEVICES', '2,3,4,5')
    def test_get_current_npu_id_visible_list(self):
        self.assertEqual(get_current_npu_id(0), 2)
        self.assertEqual(get_current_npu_id(2), 4)


class TestCpuAlloc(unittest.TestCase):

    @patch('vllm_ascend.cpu_binding.execute_command')
    def setUp(self, mock_execute_command):
        mock_execute_command.side_effect = [
            ("NPU ID  Chip ID  Chip Logic ID  Chip Name\n0 0 0 Ascend\n0 1 - Mcu\n1 0 1 Ascend",
             0),
            ("| NPU Chip | Process id |\n| 0 0 | 1234 | vllm | 56000 |\n| 1 0 | 1235 | vllm | 56000 |",
             0), ("", 0)
        ]
        self.cpu_alloc = CpuAlloc(0)

    def test_average_distribute(self):
        self.cpu_alloc.npu_cpu_pool = {
            0: [10, 11, 12, 13],
            1: [10, 11, 12, 13]
        }
        groups = {"[10, 11, 12, 13]": [0, 1]}
        result = self.cpu_alloc.average_distribute(groups)
        self.assertEqual(result, {0: [10, 11], 1: [12, 13]})
        self.cpu_alloc.npu_cpu_pool = {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            1: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            2: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        }
        groups = {"[0, 1, 2, 3, 4, 5]": [0, 1, 2]}
        result = self.cpu_alloc.average_distribute(groups)
        self.assertEqual(result, {
            0: [0, 1, 2, 3],
            1: [4, 5, 6, 7],
            2: [8, 9, 10, 11, 12, 13]
        })

    def test_extend_numa(self):
        result = self.cpu_alloc.extend_numa([])
        self.assertEqual(result, [])
        self.cpu_alloc.cpu_node = {0: 0, 1: 0, 2: 1, 3: 1}
        self.cpu_alloc.numa_to_cpu_map = {0: [0, 1], 1: [2, 3]}
        self.cpu_alloc.device_info.allowed_cpus = [0, 1, 2, 3]
        result = self.cpu_alloc.extend_numa([0, 1])
        self.assertEqual(result, [0, 1, 2, 3])
        self.cpu_alloc.device_info.allowed_cpus = [0, 1, 3]
        result = self.cpu_alloc.extend_numa([0, 1])
        self.assertEqual(result, [0, 1, 3])

    def test_get_nearest_numa_node(self):
        self.cpu_alloc.numa_to_cpu_map = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
        self.cpu_alloc.numa_distance = {0: [10, 20, 15], 1: [20, 10, 30], 2: [15, 30, 10]}
        self.assertEqual(self.cpu_alloc.get_nearest_numa_node(0), 2)
        self.assertEqual(self.cpu_alloc.get_nearest_numa_node(1), 0)
        self.cpu_alloc.numa_distance = {}
        self.assertEqual(self.cpu_alloc.get_nearest_numa_node(2), 0)

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_build_cpu_node_map(self, mock_execute_command):
        mock_execute_command.return_value = ("", 0)
        with self.assertRaises(RuntimeError):
            self.cpu_alloc.build_cpu_node_map()
        mock_execute_command.return_value = ("0 0\n1 1\n2 0\n3 1", 0)
        self.cpu_alloc.build_cpu_node_map()
        expected_cpu_node = {0: 0, 1: 1, 2: 0, 3: 1}
        expected_numa_to_cpu_map = {0: [0, 2], 1: [1, 3]}
        self.assertEqual(self.cpu_alloc.cpu_node, expected_cpu_node)
        self.assertEqual(self.cpu_alloc.numa_to_cpu_map,
                         expected_numa_to_cpu_map)

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type')
    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_build_cpu_pools_numa_uniform_fallback(self, mock_execute_command, mock_get_device_type):
        mock_get_device_type.return_value = AscendDeviceType.A2
        mock_execute_command.side_effect = [("0 0\n1 1\n2 0\n3 1", 0), ("0 0\n1 1\n2 0\n3 1", 0)]
        self.cpu_alloc.device_info.running_npu_list = [0, 1]
        self.cpu_alloc.device_info.allowed_cpus = [0, 1, 2, 3]
        self.cpu_alloc.device_info.npu_affinity = {}
        self.assertEqual(self.cpu_alloc.npu_cpu_pool, {})
        self.cpu_alloc.device_info.npu_affinity = {0: [0, 1], 1: [2, 3]}
        self.cpu_alloc.build_cpu_pools()
        self.assertEqual(len(self.cpu_alloc.npu_cpu_pool), 2)

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type')
    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_build_cpu_pools_numa_affinity_a2(self, mock_execute_command, mock_get_device_type):
        mock_get_device_type.return_value = AscendDeviceType.A2
        mock_execute_command.side_effect = [("0 0\n1 1\n2 0\n3 1", 0), ("0 0\n1 1\n2 0\n3 1", 0)]
        self.cpu_alloc.device_info.running_npu_list = [0, 1]
        self.cpu_alloc.device_info.allowed_cpus = [0, 1, 2, 3]
        self.cpu_alloc.device_info.npu_affinity = {0: [0, 2], 1: [1, 3]}
        self.cpu_alloc.build_cpu_pools()
        self.assertEqual(self.cpu_alloc.npu_cpu_pool[0], [0, 1])
        self.assertEqual(self.cpu_alloc.npu_cpu_pool[1], [2, 3])

    @patch('vllm_ascend.cpu_binding.get_ascend_device_type')
    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_build_cpu_pools_uniform_a3(self, mock_execute_command, mock_get_device_type):
        mock_get_device_type.return_value = AscendDeviceType.A3
        mock_execute_command.return_value = ("0 0\n1 0\n2 1\n3 1\n4 0\n5 1", 0)
        self.cpu_alloc.device_info.running_npu_list = [0, 1]
        self.cpu_alloc.device_info.allowed_cpus = [0, 1, 2, 3, 4, 5]
        self.cpu_alloc.device_info.npu_affinity = {0: [0, 1], 1: [2, 3]}
        self.cpu_alloc.build_cpu_pools()
        self.assertEqual(self.cpu_alloc.npu_cpu_pool[0], [0, 1, 4])
        self.assertEqual(self.cpu_alloc.npu_cpu_pool[1], [2, 3, 5])

    def test_get_nearest_numa_node_fallback_order(self):
        self.cpu_alloc.numa_to_cpu_map = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
        self.cpu_alloc.numa_distance = {}
        self.assertEqual(self.cpu_alloc.get_nearest_numa_node(1), 2)

    def test_build_cpu_pools_numa_uniform_no_allowed_cpus(self):
        self.cpu_alloc.numa_to_cpu_map = {0: [0, 1], 1: [2, 3]}
        self.cpu_alloc.device_info.allowed_cpus = []
        self.cpu_alloc.device_info.running_npu_list = [0, 1]
        self.cpu_alloc.build_cpu_pools_numa_uniform()
        self.assertEqual(self.cpu_alloc.npu_cpu_pool, {})

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_allocate(self, mock_execute_command):
        self.cpu_alloc.device_info.running_npu_list = [0]
        self.cpu_alloc.npu_cpu_pool = {0: [0, 1, 2]}
        self.cpu_alloc.allocate()
        self.assertEqual(self.cpu_alloc.assign_main[0], [0])
        self.assertEqual(self.cpu_alloc.assign_acl[0], [1])
        self.assertEqual(self.cpu_alloc.assign_rel[0], [2])
        self.cpu_alloc.npu_cpu_pool = {0: [0, 1]}
        with self.assertRaises(RuntimeError):
            self.cpu_alloc.allocate()

    @patch('vllm_ascend.cpu_binding.execute_command')
    def test_bind_threads(self, mock_execute_command):
        thread_message = "1234 1234 ? 00:00:03 acl_thread\n4567 4567 ? 00:00:03 release_thread"
        mock_execute_command.return_value = (thread_message, 0)
        self.cpu_alloc.device_info.running_npu_list = [0]
        self.cpu_alloc.assign_main = {0: [0, 1]}
        self.cpu_alloc.assign_acl = {0: [2]}
        self.cpu_alloc.assign_rel = {0: [3]}
        self.cpu_alloc.bind_threads()
        mock_execute_command.assert_called()


if __name__ == '__main__':
    unittest.main()
