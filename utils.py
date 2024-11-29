import numpy as np


def FFTConv(image: np.ndarray, kernel: np.ndarray, axis=None, mode='full'):
    if (len(image.shape) > 2):
        P1 = np.fft.rfftn(image, axes=axis)
        P2 = np.fft.rfftn(kernel, (image[0].shape))
        P2 = np.repeat([P2], image.shape[0], axis=0)
        full_conv_result = np.fft.irfftn(P1 * P2, axes=axis)
    else:
        P1 = np.fft.rfftn(image)
        P2 = np.fft.rfftn(kernel, image.shape)
        full_conv_result = np.fft.irfftn(P1 * P2)

    if mode == 'valid':
        # Calculate valid output shape
        valid_shape = tuple(np.array(image.shape) - np.array(kernel.shape) + 1)

        # Use direct slicing to extract the valid region.
        # Assuming we want to extract from the top-left corner with appropriate valid dimensions.
        slices = tuple(slice(0, end) for end in valid_shape)
        valid_result = full_conv_result[slices]
        return valid_result
    else:
        # Default is the full convolution result
        return full_conv_result


leftSobelKernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
rightSobelKernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
topSobelKernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
bottomSobelKernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
guassianKernel = 1 / 16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
outlineKernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
embossKernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
sharpenKernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])


def Conv_sharp(image, axis=None):
    # blur = FFTConv(image, guassianKernel, axis=axis)
    FFTConv_L = FFTConv(image, leftSobelKernel, axis=axis)
    FFTConv_R = FFTConv(image, rightSobelKernel, axis=axis)
    FFTConv_X = np.sqrt(np.square(FFTConv_L) + np.square(FFTConv_R))

    FFTConv_T = FFTConv(image, topSobelKernel, axis=axis)
    FFTConv_B = FFTConv(image, bottomSobelKernel, axis=axis)
    FFTConv_Y = np.sqrt(np.square(FFTConv_T) + np.square(FFTConv_B))

    FFTConv_img = np.sqrt(np.square(FFTConv_X) + np.square(FFTConv_Y))
    return FFTConv_img


def Conv_filters(image, axis=None):
    blur = FFTConv(image, guassianKernel, axis=axis)
    outline = FFTConv(image, outlineKernel, axis=axis)
    emboss = FFTConv(image, embossKernel, axis=axis)
    # edgeSharp = Conv_sharp(image, axis=axis)
    stack = np.r_[image, blur, outline, emboss]

    return stack
