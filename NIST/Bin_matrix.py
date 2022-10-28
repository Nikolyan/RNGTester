from copy import copy

class Bin_matrix:
    def __init__(self, matrix: list, rows: int, colunms: int):
        self.ROWS = rows
        self.COLUNMS = colunms
        self.matrix = matrix
        self.Low = min(rows, colunms)

    def compute_rank(self):
        i = 0
        while i < self.Low - 1:
            if self.matrix[i][i] == 1:
                self.row_operations(i, True)
            else:
                found = self.find_swap(i, True)
                if found == 1:
                    self.row_operations(i, True)
            i += 1
        i = self.ROWS - 1
        while i > 0:
            if self.matrix[i][i] == 1:
                self.row_operations(i, False)
            else:
                if self.find_swap(i, False) == 1:
                    self.row_operations(i, False)
            i -= 1
        return self.determine_rank()

    def row_operations(self, i: int, forward: bool):
        if forward:
            j = i + 1
            while j < self.ROWS:
                if self.matrix[j][i] == 1:
                    self.matrix[j, :] = (self.matrix[j, :] + self.matrix[i, :]) % 2
                j += 1
        else:
            j = i - 1
            while j >= 0:
                if self.matrix[j][i] == 1:
                    self.matrix[j, :] = (self.matrix[j, :] + self.matrix[i, :]) % 2
                j -= 1

    def find_swap(self, i, forward: bool):
        row_op = 0
        if forward:
            ind = i + 1
            while ind < self.ROWS and self.matrix[ind][i] == 0:
                ind += 1
            if ind < self.ROWS:
                row_op = self.swap_rows(i, ind)
        else:
            ind = i - 1
            while ind >= 0 and self.matrix[ind][i] == 0:
                ind -= 1
            if ind >= 0:
                row_op = self.swap_rows(i, ind)
        return row_op

    def swap_rows(self, i, ix):
        temp = copy(self.matrix[i, :])
        self.matrix[i, :] = self.matrix[ix, :]
        self.matrix[ix, :] = temp
        return 1

    def determine_rank(self):
        rank = self.ROWS
        i = 0
        while i < self.ROWS:
            all_zeros = 1
            for j in range(self.COLUNMS):
                if self.matrix[i][j] == 1:
                    all_zeros = 0
            if all_zeros == 1:
                rank -= 1
            i += 1
        return rank