import torch
import matplotlib.pyplot as plt
from src.training.data.text_encoder_decoder import encode_text_to_image_dense, decode_image_to_text_dense

def run_test(prompt, answer, image_size=(32, 32), tiny_header=False):
    header_mode = "tiny" if tiny_header else "standard"
    print(f"\n=== Running {header_mode.upper()} header test ===")

    # Encode text into image
    img = encode_text_to_image_dense(prompt, answer, image_size=image_size, tiny_header=tiny_header)

    # Visualize header pixels (first 4 pixels)
    first_pixels = (img[:, 0, 0:4] * 255).byte().permute(1, 0)
    print(f"Header pixel bytes ({header_mode}):")
    print(first_pixels.tolist())

    # Save encoded image
    save_path = f"encoded_text_image_{header_mode}.png"
    img_np = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
    plt.imsave(save_path, img_np)
    print(f"Encoded image saved to {save_path} with shape {img_np.shape}")

    # Decode back
    decoded_prompt, decoded_answer = decode_image_to_text_dense(img)

    print("Decoded Prompt:", decoded_prompt)
    print("Decoded Answer:", decoded_answer)
    print("Match Prompt:", decoded_prompt == prompt)
    print("Match Answer:", decoded_answer == answer)
    assert decoded_prompt == prompt, "Prompt mismatch"
    assert decoded_answer == answer, "Answer mismatch"
    return img

def main():
    prompt = "This is a test of dense text encoding."
    answer = "If this works, we should get the same text back."
    
    # Run both header types
    img_std = run_test(prompt, answer, image_size=(32, 32), tiny_header=False)
    img_tiny = run_test(prompt, answer, image_size=(32, 32), tiny_header=True)

    # Show combined preview
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_std.permute(1, 2, 0).numpy())
    plt.title("Standard Header")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(img_tiny.permute(1, 2, 0).numpy())
    plt.title("Tiny Header")
    plt.axis("off")
    plt.savefig("encoded_text_comparison.png")
    print("Comparison image saved to encoded_text_comparison.png")

if __name__ == "__main__":
    main()