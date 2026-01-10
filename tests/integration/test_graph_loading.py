# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Integration tests for graph loading."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from dnn_benchmarking.common.exceptions import GraphLoadError
from dnn_benchmarking.graph import GraphLoader, TensorInfo


class TestGraphLoading:
    """Integration tests for GraphLoader."""

    def test_load_sample_conv_fwd_json(self) -> None:
        """Test loading the sample conv fwd JSON file."""
        sample_path = Path(__file__).parent.parent.parent / "graphs" / "sample_conv_fwd.json"

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)

        assert graph_json["name"] == "sample_conv_fwd_16x16x16x16_k16_3x3"
        assert len(graph_json["tensors"]) == 3
        assert len(graph_json["nodes"]) == 1

    def test_validate_sample_conv_fwd(self) -> None:
        """Test validating the sample conv fwd graph."""
        sample_path = Path(__file__).parent.parent.parent / "graphs" / "sample_conv_fwd.json"

        if not sample_path.exists():
            pytest.skip(f"Sample graph not found: {sample_path}")

        loader = GraphLoader()
        graph_json = loader.load_json(sample_path)

        # Should not raise
        loader.validate(graph_json)

    def test_extract_tensor_info(self, sample_conv_fwd_json: Dict[str, Any]) -> None:
        """Test extracting tensor info from graph JSON."""
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(sample_conv_fwd_json)

        # Should have 3 non-virtual tensors
        assert len(tensor_infos) == 3

        # Check UIDs (0=output, 1=input_x, 2=weight)
        uids = {ti.uid for ti in tensor_infos}
        assert uids == {0, 1, 2}

        # Check output tensor is marked correctly
        output_tensors = [ti for ti in tensor_infos if ti.is_output]
        assert len(output_tensors) == 1
        assert output_tensors[0].uid == 0

    def test_tensor_info_size_calculation(self, sample_conv_fwd_json: Dict[str, Any]) -> None:
        """Test tensor size calculation."""
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(sample_conv_fwd_json)

        # Input tensor: [16, 16, 16, 16] float32 = 16*16*16*16*4 bytes
        input_tensor = next(ti for ti in tensor_infos if ti.uid == 1)
        assert input_tensor.dims == [16, 16, 16, 16]
        assert input_tensor.num_elements == 16 * 16 * 16 * 16
        assert input_tensor.size_bytes == 16 * 16 * 16 * 16 * 4  # float32

        # Weight tensor: [16, 16, 3, 3] float32
        weight_tensor = next(ti for ti in tensor_infos if ti.uid == 2)
        assert weight_tensor.dims == [16, 16, 3, 3]
        assert weight_tensor.num_elements == 16 * 16 * 3 * 3
        assert weight_tensor.size_bytes == 16 * 16 * 3 * 3 * 4

    def test_load_from_temp_file(
        self, temp_json_file: Path, sample_conv_fwd_json: Dict[str, Any]
    ) -> None:
        """Test loading from a temporary JSON file."""
        loader = GraphLoader()
        graph_json = loader.load_json(temp_json_file)

        assert graph_json["name"] == sample_conv_fwd_json["name"]
        assert len(graph_json["tensors"]) == len(sample_conv_fwd_json["tensors"])

    def test_load_nonexistent_file_raises(self) -> None:
        """Test that loading a nonexistent file raises GraphLoadError."""
        loader = GraphLoader()

        with pytest.raises(GraphLoadError, match="not found"):
            loader.load_json(Path("/nonexistent/path/graph.json"))

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        """Test that loading invalid JSON raises GraphLoadError."""
        invalid_json = tmp_path / "invalid.json"
        invalid_json.write_text("{ invalid json }")

        loader = GraphLoader()

        with pytest.raises(GraphLoadError, match="Invalid JSON"):
            loader.load_json(invalid_json)

    def test_get_graph_name(self, sample_conv_fwd_json: Dict[str, Any]) -> None:
        """Test getting graph name."""
        loader = GraphLoader()
        name = loader.get_graph_name(sample_conv_fwd_json)

        assert name == "sample_conv_fwd_16x16x16x16_k16_3x3"

    def test_get_graph_name_default(self) -> None:
        """Test default graph name when not specified."""
        loader = GraphLoader()
        name = loader.get_graph_name({})

        assert name == "unnamed_graph"

    def test_get_engine_id(self, sample_conv_fwd_json: Dict[str, Any]) -> None:
        """Test getting engine ID from graph."""
        loader = GraphLoader()
        engine_id = loader.get_engine_id(sample_conv_fwd_json)

        # Current fixture doesn't include preferred_engine_id (hipDNN format)
        assert engine_id is None

    def test_get_engine_id_none(self) -> None:
        """Test getting engine ID when not specified."""
        loader = GraphLoader()
        engine_id = loader.get_engine_id({})

        assert engine_id is None
