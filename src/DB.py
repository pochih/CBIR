# pylint: disable=invalid-name,missing-docstring,exec-used,too-many-arguments,too-few-public-methods,no-self-use

from __future__ import print_function

import os

import pandas as pd

DB_dir = 'database'
DB_csv = 'data.csv'


class Database:

    def __init__(self):
        self._gen_csv()
        self.data = pd.read_csv(DB_csv)
        self.classes = set(self.data["cls"])

    def _gen_csv(self):
        if os.path.exists(DB_csv):
            return
        with open(DB_csv, 'w', encoding='UTF-8') as f:
            f.write("img,cls")

            for root, _, files in os.walk(DB_dir, topdown=False):
                cls = root.split('/')[-1]

                for name in files:
                    if not name.endswith('.jpg'):
                        continue
                    img = os.path.join(root, name)
                    f.write("\n{},{}".format(img, cls))

    def __len__(self):
        return len(self.data)

    def get_class(self):
        return self.classes

    def get_data(self):
        return self.data


if __name__ == "__main__":
    db = Database()
    data = db.get_data()
    classes = db.get_class()

    print("DB length:", len(db))
    print(classes)
