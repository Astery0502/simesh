import numpy as np
import pytest
from simesh.geometry.amr.morton_order import encode_Morton, level1_Morton_order

def test_encode_Morton():
    # Test case 1: Origin point
    assert encode_Morton((0, 0, 0)) == 0
    
    # Test case 2: Simple coordinates
    assert encode_Morton((1, 0, 0)) == 1
    assert encode_Morton((0, 1, 0)) == 2
    assert encode_Morton((1, 1, 0)) == 3
    assert encode_Morton((0, 0, 1)) == 4
    
    # Test case 3: Larger coordinates
    result = encode_Morton((2, 3, 4))
    assert isinstance(result, (int, np.integer))

def test_level1_Morton_order():
    # Test case 1: 2x2x2 grid
    ig1, ig2, ig3 = 2, 2, 2
    iglevel1_sfc, sfc_iglevel1 = level1_Morton_order(ig1, ig2, ig3)
    
    # Check shapes
    assert iglevel1_sfc.shape == (2, 2, 2)
    assert sfc_iglevel1.shape == (8, 3)
    
    # Check types
    assert iglevel1_sfc.dtype == np.int32 or iglevel1_sfc.dtype == np.int64
    assert sfc_iglevel1.dtype == np.int32 or sfc_iglevel1.dtype == np.int64
    
    # Test case 2: Non-uniform grid
    ig1, ig2, ig3 = 2, 3, 4
    iglevel1_sfc, sfc_iglevel1 = level1_Morton_order(ig1, ig2, ig3)
    
    # Check shapes
    assert iglevel1_sfc.shape == (2, 3, 4)
    assert sfc_iglevel1.shape == (24, 3)

def test_invalid_inputs():
    # Test invalid input dimensions
    with pytest.raises(AssertionError):
        encode_Morton((1, 2))  # type: ignore # Only 2 dimensions
    
    with pytest.raises(AssertionError):
        encode_Morton((1, 2, 3, 4))  # type: ignore # 4 dimensions

def test_morton_order_properties():
    # Test for a small grid
    ig1, ig2, ig3 = 2, 2, 2
    iglevel1_sfc, sfc_iglevel1 = level1_Morton_order(ig1, ig2, ig3)
    
    # Test uniqueness of Morton numbers
    unique_values = np.unique(iglevel1_sfc)
    assert len(unique_values) == ig1 * ig2 * ig3
    
    # Test range of Morton numbers
    assert np.min(iglevel1_sfc) >= 0
    assert np.max(iglevel1_sfc) < ig1 * ig2 * ig3

def test_consistency():
    # Test if encoding and level1 Morton order are consistent
    ig1, ig2, ig3 = 2, 2, 2
    iglevel1_sfc, sfc_iglevel1 = level1_Morton_order(ig1, ig2, ig3)
    
    # Test a specific point
    point = (1, 1, 1)
    morton_num = encode_Morton(point)
    
    # The encoded point should exist in the Morton order grid
    assert morton_num in iglevel1_sfc

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__])