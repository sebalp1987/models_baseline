import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def show_pairs(samples_a, samples_b,
               plot_fn_a="imshow", plot_fn_b="imshow"):
    """
    Plot pairs of data
    :param samples_a:   List of samples A
    :param samples_b:   List of samples B
    :param plot_fn_a:   Name of Matplotlib function to plot A (default: "imshow")
    :param plot_fn_b:   Name of Matplotlib function to plot B (default: "imshow")
    :return:            /
    """
    assert (len(samples_a) == len(samples_b))
    num_images = len(samples_a)

    figure = plt.figure(figsize=(num_images, 2))
    grid_spec = gridspec.GridSpec(1, num_images)
    for i in range(num_images):
        grid_spec_i = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=grid_spec[i], hspace=0)

        # Drawing image A:
        ax_img = figure.add_subplot(grid_spec_i[0])
        getattr(ax_img, plot_fn_a)(samples_a[i])
        plt.gray()
        ax_img.get_xaxis().set_visible(False)
        ax_img.get_yaxis().set_visible(False)

        # Drawing image B:
        ax_img = figure.add_subplot(grid_spec_i[1])
        getattr(ax_img, plot_fn_b)(samples_b[i])
        plt.gray()
        ax_img.get_xaxis().set_visible(False)
        ax_img.get_yaxis().set_visible(False)

    # plt.tight_layout()
    plt.show()


def plot_image_grid(images, titles=None, figure=None,
                    grayscale=False, transpose=False):
    """
    Plot a grid of n x m images.
    :param images:       Images in a n x m array
    :param titles:       (opt.) List of m titles for each image column
    :param figure:       (opt.) Pyplot figure (if None, will be created)
    :param grayscale:    (opt.) Flag to draw the images in grayscale
    :param transpose:    (opt.) Flag to transpose the grid
    :return:             Pyplot figure filled with the images
    """
    num_cols, num_rows = len(images), len(images[0])
    img_ratio = images[0][0].shape[1] / images[0][0].shape[0]

    if transpose:
        vert_grid_shape, hori_grid_shape = (1, num_rows), (num_cols, 1)
        figsize = (int(num_rows * 5 * img_ratio), num_cols * 5)
        wspace, hspace = 0.2, 0.
    else:
        vert_grid_shape, hori_grid_shape = (num_rows, 1), (1, num_cols)
        figsize = (int(num_cols * 5 * img_ratio), num_rows * 5)
        hspace, wspace = 0.2, 0.

    if figure is None:
        figure = plt.figure(figsize=figsize)
    imshow_params = {'cmap': plt.get_cmap('gray')} if grayscale else {}
    grid_spec = gridspec.GridSpec(*hori_grid_shape, wspace=0, hspace=0)

    for j in range(num_cols):
        grid_spec_j = gridspec.GridSpecFromSubplotSpec(
            *vert_grid_shape, subplot_spec=grid_spec[j], wspace=wspace, hspace=hspace)

        for i in range(num_rows):
            ax_img = figure.add_subplot(grid_spec_j[i])
            # ax_img.axis('off')
            ax_img.set_yticks([])
            ax_img.set_xticks([])
            if titles is not None:
                if transpose:
                    ax_img.set_ylabel(titles[j], fontsize=25)
                else:
                    ax_img.set_title(titles[j], fontsize=15)
            ax_img.imshow(images[j][i], **imshow_params)

    figure.tight_layout()
    return figure