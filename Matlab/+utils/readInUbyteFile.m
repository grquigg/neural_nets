function data = readInUbyteFile(fileName)
    fileID = fopen(fileName, 'r');
    magic = fread(fileID, 4, 'ubit8');
    dims = magic(4);
    dimensions = fread(fileID, dims, 'uint32', 'ieee-be');
    val = prod(dimensions(2:size(dimensions)));
    data = fread(fileID, [dimensions(1), val], 'ubit8');
    fclose(fileID);
end