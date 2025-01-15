"""Tests for the ImageGenerator class."""
import datetime
import io
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from ..generator import ImageGenerator
from ..workflow import WorkflowManager


@pytest.fixture
def image_generator():
    """Create an ImageGenerator instance with a mock workflow manager."""
    workflow_manager = MagicMock(spec=WorkflowManager)
    return ImageGenerator(workflow_manager)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = Image.new('RGB', (100, 100), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()


def test_save_images_creates_date_directory_structure(image_generator, sample_image, tmp_path):
    """Test that images are saved in the correct date-based directory structure."""
    # Mock current time to ensure consistent test results
    current_time = datetime.datetime(2025, 1, 15, 15, 39)
    
    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.now.return_value = current_time
        
        # Test data
        images = [sample_image]
        count = 0
        seed = 69927
        workflow_metadata = {"test": "metadata"}
        
        # Save images
        image_paths = image_generator._save_images(
            images=images,
            output_dir=str(tmp_path),
            count=count,
            seed=seed,
            workflow_metadata=workflow_metadata
        )
        
        # Expected directory structure
        expected_dir = tmp_path / "2025" / "01" / "15" / "1539"
        expected_image_path = expected_dir / "0001_seed_69927.png"
        expected_metadata_path = expected_dir / "0001_seed_69927.json"
        
        # Verify directory structure and files
        assert expected_dir.exists()
        assert expected_image_path.exists()
        assert expected_metadata_path.exists()
        assert image_paths == [str(expected_image_path)]
        
        # Verify metadata content
        with open(expected_metadata_path) as f:
            saved_metadata = json.load(f)
            assert saved_metadata == workflow_metadata


def test_save_images_handles_multiple_images(image_generator, sample_image, tmp_path):
    """Test saving multiple images in the date-based directory structure."""
    current_time = datetime.datetime(2025, 1, 15, 15, 39)
    
    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.now.return_value = current_time
        
        # Test data for multiple images
        images = [sample_image, sample_image]  # Two identical images for testing
        count = 0
        seed = 69927
        workflow_metadata = {"test": "metadata"}
        
        # Save images
        image_paths = image_generator._save_images(
            images=images,
            output_dir=str(tmp_path),
            count=count,
            seed=seed,
            workflow_metadata=workflow_metadata
        )
        
        # Expected directory structure
        expected_dir = tmp_path / "2025" / "01" / "15" / "1539"
        expected_paths = [
            expected_dir / "0001_seed_69927.png",
            expected_dir / "0002_seed_69927.png"
        ]
        expected_metadata_paths = [
            expected_dir / "0001_seed_69927.json",
            expected_dir / "0002_seed_69927.json"
        ]
        
        # Verify all files were created
        assert expected_dir.exists()
        assert all(path.exists() for path in expected_paths)
        assert all(path.exists() for path in expected_metadata_paths)
        assert image_paths == [str(path) for path in expected_paths]
        
        # Verify metadata content for all files
        for metadata_path in expected_metadata_paths:
            with open(metadata_path) as f:
                saved_metadata = json.load(f)
                assert saved_metadata == workflow_metadata


def test_save_images_handles_errors(image_generator, tmp_path):
    """Test error handling when saving images."""
    current_time = datetime.datetime(2025, 1, 15, 15, 39)
    
    with patch('datetime.datetime') as mock_datetime:
        mock_datetime.now.return_value = current_time
        
        # Create invalid image data that will cause an error
        invalid_image = b'invalid image data'
        
        # Test data
        images = [invalid_image]
        count = 0
        seed = 69927
        workflow_metadata = {"test": "metadata"}
        
        # Save images (should not raise exception but return empty list)
        image_paths = image_generator._save_images(
            images=images,
            output_dir=str(tmp_path),
            count=count,
            seed=seed,
            workflow_metadata=workflow_metadata
        )
        
        # Verify no files were created
        expected_dir = tmp_path / "2025" / "01" / "15" / "1539"
        assert expected_dir.exists()  # Directory should still be created
        assert len(list(expected_dir.glob('*'))) == 0  # But should be empty
        assert image_paths == []  # Should return empty list
