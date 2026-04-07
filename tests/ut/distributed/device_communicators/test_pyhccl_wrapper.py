from unittest.mock import MagicMock, patch

import torch
from torch.distributed import ReduceOp

from tests.ut.base import TestBase
from vllm_ascend.distributed.device_communicators.pyhccl_wrapper import (
    Function, HCCLLibrary, aclrtStream_t, buffer_type, hcclComm_t,
    hcclDataType_t, hcclDataTypeEnum, hcclRedOp_t, hcclRedOpTypeEnum,
    hcclResult_t, hcclUniqueId)


class TestHcclUniqueId(TestBase):

    def test_construct(self):
        uid = hcclUniqueId()
        uid.internal[0] = 12
        self.assertEqual(len(uid.internal), 4108)
        self.assertEqual(uid.internal[0], 12)


class TestHcclDataTypeEnum(TestBase):

    def test_torch_dtype_mapping(self):
        expected = {
            torch.int8: hcclDataTypeEnum.hcclInt8,
            torch.uint8: hcclDataTypeEnum.hcclUint8,
            torch.int32: hcclDataTypeEnum.hcclInt32,
            torch.int64: hcclDataTypeEnum.hcclInt64,
            torch.float16: hcclDataTypeEnum.hcclFloat16,
            torch.float32: hcclDataTypeEnum.hcclFloat32,
            torch.float64: hcclDataTypeEnum.hcclFloat64,
            torch.bfloat16: hcclDataTypeEnum.hcclBfloat16,
        }

        for torch_dtype, expected_enum in expected.items():
            with self.subTest(torch_dtype=torch_dtype):
                self.assertEqual(hcclDataTypeEnum.from_torch(torch_dtype),
                                 expected_enum)

    def test_unsupported_dtype_raises(self):
        with self.assertRaises(ValueError):
            hcclDataTypeEnum.from_torch(torch.complex64)


class TestHcclRedOpTypeEnum(TestBase):

    def test_torch_reduce_op_mapping(self):
        expected = {
            ReduceOp.SUM: hcclRedOpTypeEnum.hcclSum,
            ReduceOp.PRODUCT: hcclRedOpTypeEnum.hcclProd,
            ReduceOp.MAX: hcclRedOpTypeEnum.hcclMax,
            ReduceOp.MIN: hcclRedOpTypeEnum.hcclMin,
        }

        for torch_op, expected_enum in expected.items():
            with self.subTest(torch_op=torch_op):
                self.assertEqual(hcclRedOpTypeEnum.from_torch(torch_op),
                                 expected_enum)

    def test_unsupported_op_raises(self):
        unsupported_op = "NOT_EXIST"
        with self.assertRaises(ValueError):
            hcclRedOpTypeEnum.from_torch(unsupported_op)


class TestFunction(TestBase):

    def test_construct_with_valid_args(self):
        func = Function(name="foo", restype=int, argtypes=[int, str, float])
        self.assertEqual(func.name, "foo")
        self.assertIs(func.restype, int)
        self.assertEqual(func.argtypes, [int, str, float])


class TestHCLLLibrary(TestBase):

    def setUp(self):
        super().setUp()
        self._orig_path_to_library_cache = HCCLLibrary.path_to_library_cache.copy()
        self._orig_path_to_dict_mapping = HCCLLibrary.path_to_dict_mapping.copy()
        HCCLLibrary.path_to_library_cache.clear()
        HCCLLibrary.path_to_dict_mapping.clear()

    def tearDown(self):
        HCCLLibrary.path_to_library_cache = self._orig_path_to_library_cache
        HCCLLibrary.path_to_dict_mapping = self._orig_path_to_dict_mapping
        super().tearDown()

    def test_init_with_nonexistent_so(self):
        fake_path = "/definitely/not/exist/libhccl.so"
        with self.assertRaises(OSError):
            HCCLLibrary(fake_path)

    @patch("vllm_ascend.distributed.device_communicators.pyhccl_wrapper.find_hccl_library", return_value="/fake/libhccl.so")
    @patch("vllm_ascend.distributed.device_communicators.pyhccl_wrapper.ctypes.CDLL")
    def test_init_loads_library_and_exports_functions(self, mock_cdll, _mock_find_hccl_library):
        fake_lib = MagicMock()
        for func in HCCLLibrary.exported_functions:
            setattr(fake_lib, func.name, MagicMock())
        mock_cdll.return_value = fake_lib

        lib = HCCLLibrary()

        self.assertIs(lib.lib, fake_lib)
        self.assertEqual(set(lib._funcs.keys()), {func.name for func in HCCLLibrary.exported_functions})
        mock_cdll.assert_called_once_with("/fake/libhccl.so")
        self.assertIn("/fake/libhccl.so", HCCLLibrary.path_to_library_cache)
        self.assertIn("/fake/libhccl.so", HCCLLibrary.path_to_dict_mapping)
        for func in HCCLLibrary.exported_functions:
            exported = getattr(fake_lib, func.name)
            self.assertEqual(exported.restype, func.restype)
            self.assertEqual(exported.argtypes, func.argtypes)

    @patch("vllm_ascend.distributed.device_communicators.pyhccl_wrapper.ctypes.CDLL")
    def test_init_uses_cached_functions_without_reloading(self, mock_cdll):
        fake_lib = MagicMock()
        cached_funcs = {"HcclGetErrorString": MagicMock()}
        HCCLLibrary.path_to_library_cache["/cached/libhccl.so"] = fake_lib
        HCCLLibrary.path_to_dict_mapping["/cached/libhccl.so"] = cached_funcs

        lib = HCCLLibrary("/cached/libhccl.so")

        self.assertIs(lib.lib, fake_lib)
        self.assertIs(lib._funcs, cached_funcs)
        mock_cdll.assert_not_called()

    def test_hccl_get_error_string(self):
        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {"HcclGetErrorString": MagicMock(return_value=b"HCCL internal error")}

        msg = lib.hcclGetErrorString(hcclResult_t(1))

        self.assertEqual(msg, "HCCL internal error")
        lib._funcs["HcclGetErrorString"].assert_called_once()

    def test_hccl_check(self):
        lib = HCCLLibrary.__new__(HCCLLibrary)
        mock_fn = MagicMock()
        mock_fn.return_value = "fake error"
        lib.hcclGetErrorString = mock_fn
        result = hcclResult_t(123)
        with self.assertRaises(RuntimeError) as cm:
            lib.HCCL_CHECK(result)

        self.assertEqual(str(cm.exception), "HCCL error: fake error")

    def test_hccl_check_accepts_zero(self):
        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib.hcclGetErrorString = MagicMock()

        self.assertIsNone(lib.HCCL_CHECK(0))
        lib.hcclGetErrorString.assert_not_called()

    @patch.object(HCCLLibrary, "HCCL_CHECK")
    def test_hccl_get_uniqueId(self, mock_HCCL_CHECK):
        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {"HcclGetRootInfo": MagicMock(return_value=0)}
        unique_id = lib.hcclGetUniqueId()
        self.assertIsInstance(unique_id, hcclUniqueId)
        lib._funcs["HcclGetRootInfo"].assert_called_once()
        mock_HCCL_CHECK.assert_called_once_with(0)

    @patch.object(HCCLLibrary, "HCCL_CHECK")
    def test_hccl_comm_initRank(self, mock_hccl_check):
        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {"HcclCommInitRootInfo": MagicMock(return_value=0)}

        world_size = 4
        unique_id = hcclUniqueId()
        rank = 1

        comm = lib.hcclCommInitRank(world_size, unique_id, rank)
        self.assertIsInstance(comm, hcclComm_t)
        lib._funcs["HcclCommInitRootInfo"].assert_called_once()
        mock_hccl_check.assert_called_once_with(0)

    @patch.object(HCCLLibrary, "HCCL_CHECK")
    def test_hccl_all_reduce(self, mock_hccl_check):

        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {"HcclAllReduce": MagicMock(return_value=0)}
        sendbuff = buffer_type()
        recvbuff = buffer_type()
        count = 10
        datatype = hcclDataType_t(1)
        op = hcclRedOp_t(0)
        comm = hcclComm_t()
        stream = aclrtStream_t()

        lib.hcclAllReduce(sendbuff, recvbuff, count, datatype, op, comm,
                          stream)

        lib._funcs["HcclAllReduce"].assert_called_once_with(
            sendbuff, recvbuff, count, datatype, op, comm, stream)
        mock_hccl_check.assert_called_once_with(0)

    @patch.object(HCCLLibrary, "HCCL_CHECK")
    def test_hccl_broad_cast(self, mock_hccl_check):

        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {"HcclBroadcast": MagicMock(return_value=0)}
        buff = buffer_type()
        count = 10
        datatype = 1
        root = 0
        comm = hcclComm_t()
        stream = aclrtStream_t()

        lib.hcclBroadcast(buff, count, datatype, root, comm, stream)

        lib._funcs["HcclBroadcast"].assert_called_once_with(
            buff, count, datatype, root, comm, stream)
        mock_hccl_check.assert_called_once_with(0)

    @patch.object(HCCLLibrary, "HCCL_CHECK")
    def test_hcclCommDestroy_success(self, mock_hccl_check):
        lib = HCCLLibrary.__new__(HCCLLibrary)
        lib._funcs = {"HcclCommDestroy": MagicMock(return_value=0)}
        comm = hcclComm_t()
        lib.hcclCommDestroy(comm)
        lib._funcs["HcclCommDestroy"].assert_called_once_with(comm)
        mock_hccl_check.assert_called_once_with(0)
