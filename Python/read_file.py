import struct
from array import array
if __name__ == "__main__":
    path = "mnist/train-images.idx3-ubyte"
    with open(path, 'rb') as file:
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
        print(magic)
        print(size)
        array = array("B", file.read())
        string = ""
        for i in range(600, 764):
            string += hex(array[i])
        print(string)