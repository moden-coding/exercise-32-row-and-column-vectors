#!/usr/bin/env python3

import unittest

import numpy as np

from src.row_and_column_vectors import get_column_vectors, get_row_vectors


class TestGetRowVectors(unittest.TestCase):

    def test_row_types(self):
        a = np.random.randint(0, 10, (4, 5))
        rows = get_row_vectors(a)
        self.assertIsInstance(
            rows, list,
            msg="The function get_row_vectors should return a list!")
        for row in rows:
            self.assertIsInstance(
                row, np.ndarray, msg="The list elements should be arrays!")

    def test_row_count(self):
        a = np.random.randint(0, 100, (3, 5))
        self.assertEqual(
            len(get_row_vectors(a)), 3, msg="Wrong number of rows")

    def test_row_content(self):
        n = 4
        m = 5
        a = np.random.randint(0, 10, (n, m))
        rows = get_row_vectors(a)
        for ri, row in enumerate(rows):
            self.assertEqual(row.shape, (1, m), msg="Incorrect shape!")
            for ci in range(m):
                self.assertEqual(
                    a[ri, ci], row[0, ci],
                    msg="Incorrect value at (%i,%i)!" % (ri, ci))

    def test_row_vectors_are_2d_not_1d(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        rows = get_row_vectors(a)
        self.assertEqual(
            rows[0].shape, (1, 3),
            msg="get_row_vectors should return each row as a (1, m) row "
                "vector, not a flat (m,) array. Got shape %r for row 0 of "
                "%r." % (rows[0].shape, a))


class TestGetColumnVectors(unittest.TestCase):

    def test_columns_types(self):
        a = np.random.randint(0, 10, (4, 5))
        columns = get_column_vectors(a)
        self.assertIsInstance(
            columns, list,
            msg="The function get_column_vectors should return a list!")
        for column in columns:
            self.assertIsInstance(
                column, np.ndarray, msg="The list elements should be arrays!")

    def test_column_count(self):
        a = np.random.randint(0, 100, (3, 5))
        self.assertEqual(
            len(get_column_vectors(a)), 5, msg="Wrong number of columns")

    def test_column_content(self):
        n = 4
        m = 5
        a = np.random.randint(0, 10, (n, m))
        columns = get_column_vectors(a)
        for ci, column in enumerate(columns):
            self.assertEqual(column.shape, (n, 1), msg="Incorrect shape!")
            for ri in range(n):
                self.assertEqual(
                    a[ri, ci], column[ri, 0],
                    msg="Incorrect value at (%i,%i)!" % (ri, ci))

    def test_column_vectors_are_2d_not_1d(self):
        a = np.array([[1, 2, 3], [4, 5, 6]])
        columns = get_column_vectors(a)
        self.assertEqual(
            columns[0].shape, (2, 1),
            msg="get_column_vectors should return each column as an (n, 1) "
                "column vector, not a flat (n,) array. Got shape %r for "
                "column 0 of %r." % (columns[0].shape, a))


if __name__ == '__main__':
    unittest.main()
