import rebel  # RBLN Compiler
import torch
import torchvision


def main():
    model_name = "resnet50"

    # Instantiate TorchVision model
    weights = torchvision.models.get_model_weights(model_name).DEFAULT
    model = getattr(torchvision.models, model_name)(weights=weights).eval()

    # Get input shape
    input_np = torch.rand(1, 3, 224, 224)
    input_np = weights.transforms()(input_np)

    # Compile torch model for ATOM
    input_info = [
        ("input_np", list(input_np.shape), torch.float32),
    ]
    compiled_model = rebel.compile_from_torch(model, input_info)

    # Save compiled results to disk
    compiled_model.save(f"{model_name}.rbln")


if __name__ == "__main__":
    main()
