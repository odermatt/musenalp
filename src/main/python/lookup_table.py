import numpy


class LookupTable:
    def __init__(self, values, dimensions, length=1):
        # return LookupTable(values, dimensions)
        self.values = values  # numpy array of all values in the LUT
        self.dimensions = dimensions  # array of numpy arrays of the different dimensions
        n = len(dimensions)
        self.length = length

        self.strides = numpy.ones(n)
        stride = length
        for i in range(n, 0, -1):
            self.strides[i - 1] = stride
            stride *= len(dimensions[i - 1])

        self.offset = numpy.zeros(1 << n)
        for i in range(0, len(self.strides)):
            k = 1 << i
            for j in range(0, k):
                self.offset[k + j] = self.offset[j] + self.strides[i]

    def getValue(self, coordinates):
        if coordinates is None:
            raise ValueError("array == null")

        if len(coordinates) != len(self.dimensions):
            raise ValueError(
                "array.length = " + str(len(coordinates)) + " does not correspond to the expected length " + str(
                    len(self.dimensions)))
        # TODO fracIndexes are deleted because of performance (4x better performance), not needed at the moment
        fracIndexes = numpy.zeros((len(self.dimensions), 2))
        for i in range(0, len(self.dimensions)):
           fracIndexes[i, :] = self.computeFracIndex(self.dimensions[i], coordinates[i])

        origin = 0
        for i in range(0, len(self.dimensions)):
           origin += fracIndexes[i, 0] * self.strides[i]
        if self.length == 1:
            v = numpy.zeros(1 << len(coordinates), dtype=numpy.float32)
        else:
            v = numpy.zeros((1 << len(coordinates), self.length), dtype=numpy.float32)
        for i in range(0, len(v)):
            if self.length == 1:
                v[i] = self.values[int(origin + self.offset[i])]
            else:
                numpy.copyto(v[i], self.values[int(origin + self.offset[i]):int(origin + self.offset[i] + self.length)])
        # for i in range(len(self.dimensions) - 1, -1, -1):
        #    m = 1 << i
        #    f = fracIndexes[i, 1]
        #    for j in range(0, m):
        #        if self.length == 1:
        #            v[j] += f * (v[m + j] - v[j])
        #        else:
        #            for k in range(0, len(v[j])):
        #                v[j][k] += f * (v[m + j][k] - v[j][k])

        return v[0]

    def computeFracIndex(self, partition, coordinate):
        fracIndex = numpy.zeros(2)
        lo = 0
        hi = len(partition) - 1
        while hi > lo + 1:
            m = (lo + hi) >> 1

            if coordinate < partition[m]:
                hi = m
            else:
                lo = m
        fracIndex[0] = lo
        fracIndex[1] = (coordinate - partition[lo]) / (partition[hi] - partition[lo])
        if fracIndex[1] < 0.0:
            fracIndex[1] = 0.0
        elif fracIndex[1] > 1.0:
            fracIndex[1] = 1.0
        return fracIndex

    def getValues(self, coordinates):
        if coordinates is None:
            raise ValueError("array == null")
        if len(coordinates) != len(self.dimensions):
            raise ValueError(
                "array.length = " + str(len(coordinates)) + " does not correspond to the expected length " + str(
                    len(self.dimensions)))
        n = len(coordinates)
        shape_coordinates = coordinates[0].shape
        for i in range(1, n):
            if shape_coordinates != coordinates[i].shape:
                raise ValueError("coordinate arrays are not same size")

        if self.length != 1:
            shape_coordinates = (self.length,) + shape_coordinates
        result = numpy.zeros(shape_coordinates)
        it = numpy.nditer(coordinates[0], flags=['multi_index'])
        while not it.finished:
            coord = [coordinates[0][it.multi_index]]
            for i in range(1, n):
                coord.append(coordinates[i][it.multi_index])
            values = self.getValue(coord)
            if self.length != 1:
                for i in range(0, len(values)):
                    result[i][it.multi_index] = values[i]
            else:
                result[it.multi_index] = values
            it.iternext()
        return result
