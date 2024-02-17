
def readInputFromFile (fileName)
    data = ""
    magic = 0
    size = 0
    arr = []
    dims = []
    File.open(fileName, "rb") do |file|
        magic = file.read(4)
        vals = magic.unpack("C*")
        data = file.read(4*vals[3])
        dims = data.unpack("L>*")
        input = file.read
        arr = input.unpack("C*")
    end
return dims, arr 
end

def main
    puts "Hello World!";
    input_path = "../mnist/train-images.idx3-ubyte"
    output_path = "../mnist/train-labels.idx1-ubyte"
    inputdims, input_arr = readInputFromFile(input_path)
    inputs = input_arr.each_slice(inputdims[1]*inputdims[2]).to_a
    outputdims, outputs = readInputFromFile(output_path)
    puts inputs.length
    puts outputs.length
end

main