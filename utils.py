from torchvision.utils import make_grid, save_image


def save_images(path_to_save, data, image_dim, nrow=6):
    img = make_grid(data.reshape([data.shape[0], *image_dim]), nrow=nrow)
    save_image(img, path_to_save)