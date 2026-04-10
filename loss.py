import torch
import torch.nn as nn
from torchvision import models, transforms


class PerceptualAndStyleLoss(nn.Module):
    """
    Perceptual and Style Loss using pre-trained VGG16 model.
    Combines content loss and style loss for training.
    """

    def __init__(self, style_layers, content_layers, img_size=224):
        """
        Args:
            style_layers: List of VGG16 layer names (e.g., 'relu1_2') for style loss
            content_layers: List of VGG16 layer names (e.g., 'relu4_2') for content loss
            img_size: Target size to resize images before VGG processing
        """
        super(PerceptualAndStyleLoss, self).__init__()

        self.style_layers = style_layers
        self.content_layers = content_layers
        self.img_size = img_size
        self.criterion = nn.MSELoss()

        # VGG16 layer names mapping
        self.layer_names_map = {
            "2": "relu1_2",
            "7": "relu2_2",
            "14": "relu3_3",
            "19": "relu4_2",
            "21": "relu4_3",
            "24": "relu5_1",
            "28": "relu5_3",
        }

        # Load pre-trained VGG16 features and freeze parameters
        self.vgg = models.vgg16(pretrained=True).features
        for param in self.vgg.parameters():
            param.requires_grad_(False)

        # Determine the maximum layer index needed
        self.max_layer_index = -1
        for idx_str, name in self.layer_names_map.items():
            if name in style_layers or name in content_layers:
                self.max_layer_index = max(self.max_layer_index, int(idx_str))

        # Trim VGG to only include layers up to the highest required one
        self.feature_extractor = nn.Sequential(
            *list(self.vgg.children())[: self.max_layer_index + 1]
        )

        # Preprocessing transform
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
            ]
        )

    def _get_features(self, image_tensor):
        """
        Extract features from specified VGG layers.

        Args:
            image_tensor: Input tensor [B, C, H, W]

        Returns:
            Dictionary mapping layer names to feature maps
        """
        processed_image = (
            torch.stack([self.transform(img) for img in image_tensor.squeeze(1)])
            if image_tensor.dim() == 5
            else self.transform(image_tensor)
        )

        features = {}
        x = processed_image
        for name, layer in self.feature_extractor._modules.items():
            x = layer(x)
            if name in self.layer_names_map:
                layer_name_str = self.layer_names_map[name]
                if (
                    layer_name_str in self.style_layers
                    or layer_name_str in self.content_layers
                ):
                    features[layer_name_str] = x
        return features

    def gram_matrix(self, input):
        """
        Compute the Gram matrix for style representation.

        Args:
            input: Feature map tensor [B, C, H, W]

        Returns:
            Gram matrix [B*C, B*C]
        """
        a, b, c, d = input.size()
        features = input.view(a * b, c * d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

    def forward(self, generated_image, target_content_image, target_style_image):
        """
        Compute perceptual and style losses.

        Args:
            generated_image: Output from generator [B, C, H, W]
            target_content_image: Ground truth content image [B, C, H, W]
            target_style_image: Reference style image [B, C, H, W]

        Returns:
            List [content_loss, style_loss]
        """
        # Extract features for all images
        gen_features = self._get_features(generated_image)
        content_features = self._get_features(target_content_image)
        style_features = self._get_features(target_style_image)

        # Calculate Content Loss
        content_loss = 0.0
        for layer_name in self.content_layers:
            content_loss += self.criterion(
                gen_features[layer_name], content_features[layer_name].detach()
            )

        # Calculate Style Loss
        style_loss = 0.0
        for layer_name in self.style_layers:
            gen_gram = self.gram_matrix(gen_features[layer_name])
            target_gram = self.gram_matrix(style_features[layer_name].detach())
            style_loss += self.criterion(gen_gram, target_gram)

        return [content_loss, style_loss]
