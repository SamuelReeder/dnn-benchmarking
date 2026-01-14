"""Tensor information dataclass."""

from dataclasses import dataclass, field
from typing import List


# Map data type strings to byte sizes
DTYPE_SIZES = {
    "float": 4,
    "half": 2,
    "bfloat16": 2,
    "double": 8,
    "int8": 1,
    "int32": 4,
    "uint8": 1,
}


@dataclass
class TensorInfo:
    """Information about a tensor extracted from graph JSON.

    Attributes:
        uid: Unique identifier for the tensor.
        name: Human-readable name of the tensor.
        dims: Dimensions of the tensor (e.g., [N, C, H, W]).
        strides: Memory strides for each dimension.
        data_type: Data type as string (e.g., "float", "half").
        is_virtual: Whether this is a virtual (intermediate) tensor.
        is_output: Whether this tensor is marked as a graph output.
    """

    uid: int
    name: str
    dims: List[int]
    strides: List[int]
    data_type: str
    is_virtual: bool
    is_output: bool = False

    @property
    def element_size(self) -> int:
        """Get size of one element in bytes."""
        return DTYPE_SIZES.get(self.data_type.lower(), 4)

    @property
    def num_elements(self) -> int:
        """Get total number of elements."""
        result = 1
        for dim in self.dims:
            result *= dim
        return result

    @property
    def size_bytes(self) -> int:
        """Get total size in bytes."""
        return self.num_elements * self.element_size

    @classmethod
    def from_json(cls, tensor_json: dict, is_output: bool = False) -> "TensorInfo":
        """Create TensorInfo from a JSON tensor object.

        Args:
            tensor_json: Dictionary containing tensor attributes from graph JSON.
            is_output: Whether this tensor is a graph output.

        Returns:
            TensorInfo instance.
        """
        return cls(
            uid=tensor_json["uid"],
            name=tensor_json.get("name", f"tensor_{tensor_json['uid']}"),
            dims=tensor_json["dims"],
            strides=tensor_json.get("strides", []),
            data_type=tensor_json.get("data_type", "float"),
            is_virtual=tensor_json.get("virtual", False),
            is_output=is_output,
        )
