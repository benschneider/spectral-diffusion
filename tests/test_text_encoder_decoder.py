import torch

from src.training.data.text_encoder_decoder import encode_text_to_image_dense


def test_encode_text_to_image_dense_deterministic():
    img1 = encode_text_to_image_dense("Prompt", "Answer", image_size=(16, 16))
    img2 = encode_text_to_image_dense("Prompt", "Answer", image_size=(16, 16))
    assert img1.shape == (3, 16, 16)
    assert torch.allclose(img1, img2)
    assert img1.min().item() >= 0.0
    assert img1.max().item() <= 1.0


def test_encode_text_to_image_dense_varies_with_input():
    img1 = encode_text_to_image_dense("Prompt A", "Answer", image_size=(16, 16))
    img2 = encode_text_to_image_dense("Prompt B", "Answer", image_size=(16, 16))
    assert not torch.allclose(img1, img2)
