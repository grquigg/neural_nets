
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

def initializeRandomWeights(height, width)
    weights = Array.new(height) {Array.new(width, 0)}
    (0..height-1).each do |i|
        (0..width-1).each do |j|
            weights[i][j] = rand(0.0..1.0)
        end
    end
return weights
end

def dotProduct(matrix1, matrix2)
    i = 0
    j = 0
    k = 0
    product = Array.new(matrix1.length) {Array.new(matrix2[0].length, 0.0)}
    # puts matrix1.length
    # puts matrix1[0].length
    # puts matrix2.length
    # puts matrix2[0].length
    while i < matrix1.length do
        j = 0
        while j < matrix2[0].length do
            (0..matrix1[0].length-1).each { |k| product[i][j] += (matrix1[i][k] * matrix2[k][j])}
            j += 1
        end
        i += 1
    end
return product
end

def computeAccuracy(predicted, actual)
    total = 0
    (0..predicted.length-1).each do |i|
        max_index = 0
        max_val = 0.0
        (0..predicted[i].length-1).each do |j|
            if predicted[i][j] > max_val
                max_val = predicted[i][j]
                max_index = j
            end
        end
        if actual[i] == max_index
            total += 1
        end
    end
return total
end
def printMatrix(matrix)
    (0..matrix.length-1).each do |i|
        string = ""
        (0..matrix[0].length-1).each do |j|
            string += matrix[i][j].to_s + "\t"
        end
        string += "\n"
        puts string
    end
end


def matrixSubtract(mat1, mat2)
    result = Array.new(mat1.length) {Array.new(mat1[0].length, 0)}
    (0..mat1.length-1).each {|row| (0..mat1[row].length-1).each {|col| result[row][col] = mat1[row][col] - mat2[row][col]}}
return result
end

def softmax(matrix)
    (0..matrix.length-1).each do |row|
        total = 0
        matrix[row].each do |col|
            total += Math.exp(col)
        end
        logSum = Math.log(total)
        (0..matrix[row].length-1).each do |j|
            matrix[row][j] = Math.exp(matrix[row][j] - logSum)
        end
    end
end

def normalizeInputs(inputs)
    i = 0
    while i < inputs.length do
        j = 0
        while j < inputs[0].length do
            inputs[i][j] = inputs[i][j].to_f / 255.0
            j+=1
        end
        i+=1
    end
end

def main
    puts "Hello World!";
    size = 60000
    width = 28
    height = 28
    epochs = 1
    learning_rate = 0.005
    batch_size = 1000
    input_path = "../mnist/train-images.idx3-ubyte"
    output_path = "../mnist/train-labels.idx1-ubyte"
    inputdims, input_arr = readInputFromFile(input_path)
    inputs = input_arr.each_slice(inputdims[1]*inputdims[2]).to_a
    outputdims, outputs = readInputFromFile(output_path)
    puts inputs.length
    puts outputs.length
    #normalize input values
    normalizeInputs(inputs)
    output_one_hot = Array.new(60000) {Array.new(10, 0)}
    index = 0
    while index < output_one_hot.length do
        output_one_hot[index][outputs[index]] += 1
        index += 1
    end
    weights = Array.new(width*height) {Array.new(10, 1.0)}
    # printMatrix(weights)
    (1..epochs).each do |i|
        index = 0
        num_correct = 0
        while index < size do
            product = dotProduct(inputs[index..index+batch_size], weights)
            softmax(product)
            puts "New product"
            printMatrix(product)
            num_correct += computeAccuracy(inputs[index..index+batch_size], outputs[index..index+batch_size])
            accuracy = num_correct.to_f / (index+batch_size).to_f
            puts "Accuracy: " + accuracy.to_s
            index += batch_size
            product = matrixSubtract(output_one_hot[index..index+batch_size], product)
            printMatrix(product)
            break
        end
    end
end

main