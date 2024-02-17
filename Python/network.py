import struct
from array import array
import numpy as np

def read_file(path):
    inputs = None
    with open(path, 'rb') as file:
        magic = struct.unpack("4B", file.read(4))
        dims = struct.unpack(f">{magic[3]}I", file.read(4*magic[3]))
        arr = array("B", file.read())
        arr = np.array(arr)
        inputs = np.reshape(arr, dims)
    return inputs

if __name__ == "__main__":
    input_path = "../mnist/train-images.idx3-ubyte"
    output_path = "../mnist/train-labels.idx1-ubyte"
    inputs = read_file(input_path)
    outputs = read_file(output_path)
    print(inputs.shape)
    print(outputs.shape)