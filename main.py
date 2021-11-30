import numpy as np
import pywt
import math
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import (
    erosion, dilation, closing, opening, area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table
from skimage.filters import threshold_otsu
import pandas as pd
from PIL import Image, ImageDraw
import json
from scipy.fftpack import dct, idct

square = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])


def process_image(file_name):
    img = imread(f'./images/{file_name}')
    return img.astype("float")


def multi_dil(im, num, element=square):
    for i in range(num):
        im = dilation(im, element)
    return im


def multi_ero(im, num, element=square):
    for i in range(num):
        im = erosion(im, element)
    return im


def generate_threshold_mask(img):
    threshold = threshold_otsu(img)
    result = (img > threshold)
    multi_dilated = multi_dil(result, 7)
    area_closed = area_closing(multi_dilated, 50000)
    multi_eroded = multi_ero(area_closed, 7)
    opened = opening(multi_eroded)
    return opened


def extract_region_features(binarized):
    label_im = label(binarized)
    regions = regionprops(label_im)
    properties = ['area', 'bbox_area',
                  'solidity', 'eccentricity', 'perimeter']
    table = pd.DataFrame(regionprops_table(
        label_im, binarized, properties=properties))
    table["area_ratio"] = table["area"] / table["bbox_area"]
    return (table, regions)
# ------------------------------ ROI --------------------------------


def extract_ROI(table, regions, img):
    ROI_index = table["area"].idxmax()
    box = regions[ROI_index].bbox
    return (box, ROI_index)


def extract_ROI_region(filename):
    with open(f"./images/{filename}.json", "r") as file:
        coords = json.load(file)
    box = coords["ROI"]
    return box


def process_coefficients(imArray, model, level):
    coeffs = pywt.wavedec2(data=imArray, wavelet=model, level=level)
    coeffs_H = (coeffs)
    return coeffs_H


def quantization_step(LH2, HL2, HH2, k):
    LH2 = np.absolute(LH2).sum()
    HL2 = np.absolute(HL2).sum()
    HH2 = np.absolute(HH2).sum()
    return k * round(math.log((LH2 + HL2 + HH2) / 2))


def DWT(ROI, model, level, k):
    coeffs_image = process_coefficients(ROI, model, level=level)
    LL2, (LH2, HL2, HH2), (LH1, HL1, HH1) = coeffs_image
    Q = quantization_step(LH2, HL2, HH2, k)
    print(Q)
    Xq = LL2 / Q
    return Xq, (LH2, HL2, HH2), (LH1, HL1, HH1), Q


def ROI_watermark_generation(height, width, table, index):
    watermark = Image.new("RGB", (width, height), color=(255, 255, 255))
    area = table.iloc[index].area
    area_ratio = table.iloc[index].area_ratio
    solidity = table.iloc[index].solidity
    perimeter = table.iloc[index].perimeter
    eccentricity = table.iloc[index].eccentricity
    area_ratio_str = "{:.6f}".format(area_ratio)
    solidity_str = "{:.6f}".format(solidity)
    perimeter_str = "{:.6f}".format(perimeter)
    eccentricity_str = "{:.6f}".format(eccentricity)
    watermark_text = f"{area}\n{area_ratio_str}\n{solidity_str}\n{perimeter_str}\n{eccentricity_str}"
    watermark_draw = ImageDraw.Draw(watermark)
    watermark_draw.multiline_text(
        (1, 1), watermark_text, spacing=1, fill=(0, 0, 0))
    watermark_array = np.asarray(watermark)
    watermark = rgb2gray(watermark_array)
    return watermark


def ROI_watermark_encryption(watermark):
    row, col = watermark.shape
    N = row * col
    x = 0.9 + np.zeros(N)
    r = 3.95
    for n in range(N-1):
        x[n+1] = r * x[n] * (1 - x[n])
    x = x.reshape(row, col)
    boolean_x = np.around(x).astype("int")
    watermark = watermark.astype("int")
    encrypted_watermark = np.bitwise_xor(watermark, boolean_x).astype("float")
    return encrypted_watermark, boolean_x


def ROI_watermark_embed(Xq, ROI_encrypted_watermark, Q):
    Xq_copy = Xq.copy()
    row, col = Xq_copy.shape
    for i in range(row):
        for j in range(col):
            if(round(Xq[i, j] % 2) == ROI_encrypted_watermark[i, j]):
                Xq_copy[i, j] = Xq[i, j] * Q
            else:
                Xq_copy[i, j] = (Xq[i, j] * Q) + Q
    return Xq_copy


def extract_ROI_watermark(image, boolean_x, model, level, k):
    Zq, (LH2, HL2, HH2), (LH1, HL1, HH1), Q = DWT(image, model, level, k)
    Zq = np.round(Zq)
    Wsroi = Zq % 2
    Wroi_salt = Wsroi.copy().astype("int") ^ boolean_x
    return Wroi_salt.astype("float")

# --------------------------------- RONI --------------------------------


def extract_RONI_region(filename):
    with open(f"./images/{filename}.json", "r") as file:
        coords = json.load(file)
    box = coords["RONI"]
    return box


def extract_capacity(filename):
    with open(f"./images/{filename}.json", "r") as file:
        coords = json.load(file)
    capacity = coords["capacity"]
    return capacity


def RONI_watermark_generation(capacity):
    RONI_watermark = Image.new(
        "RGB", (capacity, capacity), color=(255, 255, 255))
    RONI_watermark_text = f"Institute\nPatientSex\nBirthDate\nhdu2014"
    RONI_watermark_draw = ImageDraw.Draw(RONI_watermark)
    RONI_watermark_draw.multiline_text(
        (1, 1), RONI_watermark_text, spacing=1, fill=(0, 0, 0))
    RONI_watermark_array = np.asarray(RONI_watermark)
    RONI_watermark = rgb2gray(RONI_watermark_array)
    return RONI_watermark


def arnold_transform(watermark):
    A = np.array([[1, 1], [1, 2]])
    watermark_copy = watermark
    N = 10
    for n in range(0, N):
        watermark_final = np.zeros(watermark_copy.shape)
        for i in range(watermark_copy.shape[0]):
            for j in range(watermark_copy.shape[1]):
                x = np.array([[i], [j]])
                y = np.matmul(A, x) % watermark_copy.shape[0]
                watermark_final[y[0][0], y[1][0]] = watermark_copy[i, j]
        watermark_copy = watermark_final
    return watermark_final, N


def arnold_transform_reverse(watermark, N):
    A = np.array([[2, -1], [-1, 1]])
    watermark_final = watermark
    for n in range(0, N):
        watermark_copy = np.zeros(watermark_final.shape)
        for i in range(watermark_copy.shape[0]):
            for j in range(watermark_copy.shape[1]):
                x = np.array([[i], [j]])
                y = np.matmul(A, x) % watermark_copy.shape[0]
                watermark_copy[y[0][0], y[1][0]] = watermark_final[i, j]
        watermark_final = watermark_copy
    return watermark_final


def extract_blocks(RONI):
    block_size = 8
    i = 0
    RONI_blocks = np.empty(
        (int(RONI.shape[0]/8), int(RONI.shape[1]/8), 8, 8), dtype="uint8")
    for r in range(0, RONI.shape[0], block_size):
        for c in range(0, RONI.shape[1], block_size):
            window = RONI[r:r+block_size, c:c+block_size]
            RONI_blocks[int(r/8), int(c/8)] = window
    return RONI_blocks


def reconstruct_blocks(RONI_blocks):
    block_size = 8
    reconstructed_RONI = np.empty(
        (RONI_blocks.shape[0]*8, RONI_blocks.shape[1]*8))
    for r in range(0, reconstructed_RONI.shape[0], block_size):
        for c in range(0, reconstructed_RONI.shape[1], block_size):
            reconstructed_RONI[r:r+block_size, c:c +
                               block_size] = RONI_blocks[int(r/8), int(c/8)]
    return reconstructed_RONI


def zigzag(n):
    '''zigzag rows'''
    def compare(xy):
        x, y = xy
        return (x + y, -y if (x + y) % 2 else y)
    xs = range(n)
    return {index: n for n, index in enumerate(sorted(
        ((x, y) for x in xs for y in xs),
        key=compare
    ))}


def embed_RONI_watermark(img, zigzag_pattern, watermark_bits):
    DCT = dct(img, norm="ortho")
    zigzag_dct = np.empty(64)
    Q = 4
    for i in range(0, 64):
        zigzag_dct[i] = DCT[zigzag_pattern[i][0], zigzag_pattern[i][1]]
    middle_frequency = zigzag_dct[12:20]
    middle_frequency = np.round(middle_frequency / Q)
    embedded_frequency = np.empty(len(middle_frequency))
    for i in range(len(middle_frequency)):
        if((middle_frequency[i] + watermark_bits[i]) % 2 == 0):
            embedded_frequency[i] = (middle_frequency[i] + 0.5)*Q
        elif((middle_frequency[i] + watermark_bits[i]) % 2 == 1):
            embedded_frequency[i] = (middle_frequency[i] - 0.5)*Q
    zigzag_dct[12:20] = embedded_frequency
    inverse_zigzag = np.empty((8, 8))
    for i in range(0, 64):
        inverse_zigzag[zigzag_pattern[i][0],
                       zigzag_pattern[i][1]] = zigzag_dct[i]
    return np.round(np.absolute(idct(inverse_zigzag)))


def embed_RONI(RONI_blocks, RONI_encrypted_watermark):
    watermark_bit_count = 0
    watermark = RONI_encrypted_watermark.copy().flatten()
    zigzag_pattern = list(zigzag(8).keys())
    embedded_RONI_blocks = np.empty(RONI_blocks.shape)
    for i in range(0, RONI_blocks.shape[0]):
        for j in range(0, RONI_blocks.shape[1]):
            watermark_bits = watermark[watermark_bit_count:watermark_bit_count+8]
            embedded_RONI_blocks[i, j] = embed_RONI_watermark(
                RONI_blocks[i, j], zigzag_pattern, watermark_bits)
            watermark_bit_count += 8
    return embedded_RONI_blocks


def extract_RONI_watermark(img, zigzag_pattern):
    Q = 4
    zigzag_dct = np.empty(64)
    for i in range(0, 64):
        zigzag_dct[i] = dct(img)[zigzag_pattern[i][0], zigzag_pattern[i][1]]
    return np.floor(zigzag_dct[12:20] / Q) % 2


def extract_RONI(embedded_RONI, capacity):
    extracted_watermark = np.empty(capacity*capacity)
    watermark_bit_count = 0
    embedded_RONI_blocks = extract_blocks(embedded_RONI)
    zigzag_pattern = list(zigzag(8).keys())
    for i in range(0, embedded_RONI_blocks.shape[0]):
        for j in range(0, embedded_RONI_blocks.shape[1]):
            watermark_bits = extract_RONI_watermark(
                embedded_RONI_blocks[i, j], zigzag_pattern)
            extracted_watermark[watermark_bit_count:watermark_bit_count +
                                8] = watermark_bits
            watermark_bit_count += 8
    return np.reshape(extracted_watermark, (-1, capacity))

# -------------------------- MAIN ---------------------------------------


def embedding(filename):
    level = 2
    model = "haar"
    k = 8
    capacity = extract_capacity(filename)
    image_array = process_image(f"{filename}.png")
    binarized = generate_threshold_mask(image_array)
    table, regions = extract_region_features(binarized)
    ROI__cords, ROI_index = extract_ROI(table, regions, image_array)
    ROI = image_array[ROI__cords[0]:ROI__cords[2], ROI__cords[1]:ROI__cords[3]]
    Xq, (LH2, HL2, HH2), (LH1, HL1, HH1), Q = DWT(ROI, model, level, k)
    watermark_height, watermark_width = Xq.shape
    watermark = ROI_watermark_generation(
        watermark_height, watermark_width, table, ROI_index)
    encrypted_watermark, boolean_x = ROI_watermark_encryption(watermark)
    Xq_copy = ROI_watermark_embed(Xq, encrypted_watermark, Q)
    coeffs = [Xq_copy, (LH2, HL2, HH2), (LH1, HL1, HH1)]
    embedded_ROI = pywt.waverec2(coeffs, model)
    embedded_ROI = np.round(np.absolute(embedded_ROI))

    RONI_coords = extract_RONI_region(filename)
    RONI = image_array[RONI_coords["0"]:RONI_coords["2"],
                       RONI_coords["1"]:RONI_coords["3"]]
    RONI_watermark = RONI_watermark_generation(capacity)
    RONI_encrypted_watermark, N = arnold_transform(RONI_watermark)
    RONI_blocks = extract_blocks(RONI)
    embedded_RONI_blocks = embed_RONI(RONI_blocks, RONI_encrypted_watermark)
    reconstrcuted_RONI = reconstruct_blocks(embedded_RONI_blocks)

    image_array[ROI__cords[0]:ROI__cords[2], ROI__cords[1]:ROI__cords[3]
                ] = np.delete(embedded_ROI, embedded_ROI.shape[0]-1, 0)
    image_array[RONI_coords["0"]:RONI_coords["2"],
                RONI_coords["1"]:RONI_coords["3"]] = reconstrcuted_RONI
    return image_array, boolean_x, N


def extracting(filename, image, boolean_x, N):
    level = 2
    model = "haar"
    k = 8
    capacity = extract_capacity(filename)
    ROI_coords = extract_ROI_region(filename)
    ROI = image[ROI_coords["0"]:ROI_coords["2"],
                ROI_coords["1"]:ROI_coords["3"]]
    extracted_ROI_watermark = extract_ROI_watermark(
        ROI, boolean_x, model, level, k)

    RONI_coords = extract_RONI_region(filename)
    RONI = image[RONI_coords["0"]:RONI_coords["2"],
                 RONI_coords["1"]:RONI_coords["3"]]
    extracted_RONI_watermark = extract_RONI(RONI, capacity)
    RONI_decrypted_watermark = arnold_transform_reverse(
        extracted_RONI_watermark, N)
    return extracted_ROI_watermark, RONI_decrypted_watermark
