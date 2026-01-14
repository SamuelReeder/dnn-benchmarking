"""Validation module for comparing execution output against reference.

Currently STUBBED - CPU reference plugin not available in Python.
"""

from typing import Optional, Tuple

import numpy as np

from ..graph.tensor_info import TensorInfo


class Validator:
    """Validates execution output against a reference using allclose comparison.

    Currently STUBBED - CPU reference plugin not available in Python.
    When implemented, this will:
    1. Copy output buffer to host
    2. Compare against reference data using np.allclose(rtol, atol)
    """

    def __init__(
        self,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> None:
        """Initialize validator with tolerance settings.

        Args:
            rtol: Relative tolerance for allclose comparison.
            atol: Absolute tolerance for allclose comparison.
        """
        self._rtol = rtol
        self._atol = atol

    def validate(
        self,
        output_data: np.ndarray,
        tensor_info: TensorInfo,
        reference_data: Optional[np.ndarray] = None,
    ) -> Tuple[bool, str]:
        """Compare output data against reference using np.allclose.

        Args:
            output_data: Output tensor data from execution.
            tensor_info: Information about the output tensor.
            reference_data: Golden/reference data to compare against.
                           If None, validation is skipped.

        Returns:
            Tuple of (passed: bool, message: str).
        """
        if reference_data is None:
            return (True, "Validation skipped - no reference data provided")

        # Ensure shapes match
        if output_data.shape != reference_data.shape:
            return (
                False,
                f"Shape mismatch: output {output_data.shape} vs reference {reference_data.shape}",
            )

        # Perform allclose comparison
        passed = np.allclose(output_data, reference_data, rtol=self._rtol, atol=self._atol)

        if passed:
            return (True, f"Validation passed (rtol={self._rtol}, atol={self._atol})")
        else:
            # Calculate max difference for error message
            abs_diff = np.abs(output_data - reference_data)
            max_abs_diff = float(np.max(abs_diff))
            max_rel_diff = float(np.max(abs_diff / (np.abs(reference_data) + 1e-10)))

            return (
                False,
                f"Validation failed: max_abs_diff={max_abs_diff:.2e}, "
                f"max_rel_diff={max_rel_diff:.2e} "
                f"(rtol={self._rtol}, atol={self._atol})",
            )

    def validate_stub(self) -> Tuple[bool, str]:
        """Stubbed validation - returns success with message.

        Use this when no reference data is available (MVP).

        Returns:
            Tuple of (True, stub message).
        """
        return (True, "Validation stubbed - CPU reference not available in Python")

    def compare_ab(
        self, output_a: np.ndarray, output_b: np.ndarray
    ) -> Tuple[bool, str]:
        """Compare A and B outputs using np.allclose.

        Args:
            output_a: Output tensor data from configuration A.
            output_b: Output tensor data from configuration B.

        Returns:
            Tuple of (passed: bool, message: str).
        """
        # Check for NaN/Inf in outputs
        if np.any(np.isnan(output_a)) or np.any(np.isinf(output_a)):
            return (False, "Output A contains NaN or Inf values")

        if np.any(np.isnan(output_b)) or np.any(np.isinf(output_b)):
            return (False, "Output B contains NaN or Inf values")

        # Ensure shapes match
        if output_a.shape != output_b.shape:
            return (
                False,
                f"Shape mismatch: A={output_a.shape} vs B={output_b.shape}",
            )

        # Perform allclose comparison
        passed = np.allclose(output_a, output_b, rtol=self._rtol, atol=self._atol)

        if passed:
            return (True, f"A/B outputs match (rtol={self._rtol}, atol={self._atol})")
        else:
            # Calculate differences for error message
            abs_diff = np.abs(output_a - output_b)
            max_abs_diff = float(np.max(abs_diff))
            max_rel_diff = float(np.max(abs_diff / (np.abs(output_b) + 1e-10)))

            return (
                False,
                f"A/B mismatch: max_abs_diff={max_abs_diff:.2e}, "
                f"max_rel_diff={max_rel_diff:.2e} "
                f"(rtol={self._rtol}, atol={self._atol})",
            )
