from datasketch import MinHash, MinHashLSH
import numpy as np
import pandas as pd
from datetime import datetime
import time
from sklearn.neighbors import NearestNeighbors


class Node:
    def __init__(self, is_leaf=True):
        self.is_leaf = is_leaf
        self.children = []
        self.bounding_box = None
        self.data = None

    def update_bounding_box(self):
        if not self.children:
            return
        dimensions = len(self.children[0].bounding_box)
        min_coords = [float('inf')] * dimensions
        max_coords = [float('-inf')] * dimensions

        for child in self.children:
            for dim in range(dimensions):
                min_coords[dim] = min(min_coords[dim], child.bounding_box[dim][0])
                max_coords[dim] = max(max_coords[dim], child.bounding_box[dim][1])

        self.bounding_box = [(min_coords[d], max_coords[d]) for d in range(dimensions)]


class RTree:
    def __init__(self, max_children=4, dimensions=4):
        self.root = Node()
        self.max_children = max_children
        self.dimensions = dimensions

    def insert(self, bounding_box, data=None):
        new_node = Node(is_leaf=True)
        new_node.bounding_box = bounding_box
        new_node.data = data
        self._insert(self.root, new_node)

    def _insert(self, parent, new_node):
        if parent.is_leaf:
            parent.children.append(new_node)
            parent.update_bounding_box()
            if len(parent.children) > self.max_children:
                self._split(parent)
        else:
            best_child = min(
                parent.children,
                key=lambda child: self._enlargement(child.bounding_box, new_node.bounding_box)
            )
            self._insert(best_child, new_node)
            parent.update_bounding_box()

    def _split(self, node):
        children = node.children
        node.children = []
        group1, group2 = self._quadratic_split(children)

        node.children = group1
        node.is_leaf = group1[0].is_leaf
        node.update_bounding_box()

        new_node = Node(is_leaf=node.is_leaf)
        new_node.children = group2
        new_node.update_bounding_box()

        if node == self.root:
            new_root = Node(is_leaf=False)
            self.root = new_root
            new_root.children = [node, new_node]
        else:
            parent = self._find_parent(self.root, node)
            parent.children.append(new_node)
            parent.update_bounding_box()

    def _find_parent(self, current, target):
        if current.is_leaf:
            return None
        for child in current.children:
            if child == target or (not child.is_leaf and self._find_parent(child, target)):
                return current
        return None

    def _quadratic_split(self, children):
        group1, group2 = [], []
        sorted_children = sorted(children, key=lambda c: sum(d[0] for d in c.bounding_box))
        mid = len(sorted_children) // 2
        group1, group2 = sorted_children[:mid], sorted_children[mid:]
        return group1, group2

    def _enlargement(self, box1, box2):
        enlarged_box = [
            (min(box1[d][0], box2[d][0]), max(box1[d][1], box2[d][1]))
            for d in range(len(box1))
        ]
        return self._area(enlarged_box) - self._area(box1)

    def _area(self, box):
        area = 1
        for dim in box:
            area *= (dim[1] - dim[0])
        return area

    def search(self, query_bounds):
        results = []
        self._search(self.root, query_bounds, results)
        return results

    def _search(self, node, query_bounds, results):
        if self._intersects(node.bounding_box, query_bounds):
            if node.is_leaf:
                for child in node.children:
                    if self._intersects(child.bounding_box, query_bounds):
                        results.append(child)
            else:
                for child in node.children:
                    self._search(child, query_bounds, results)

    def _intersects(self, box1, box2):
        return all(box1[d][1] >= box2[d][0] and box1[d][0] <= box2[d][1] for d in range(len(box1)))


def load_data(file_path):
    df = pd.read_csv(file_path)
    base_date = datetime(2017, 1, 1)
    df['review_date'] = pd.to_datetime(df['review_date'], format='%B %Y')
    df['review_date'] = (df['review_date'] - base_date).dt.days
    return df


def insert_into_rtree(rtree, df):
    for _, row in df.iterrows():
        loc_country = row.get("loc_country", "")
        if not loc_country:
            continue
        rating = row["rating"]
        review_date = row["review_date"]
        usd_100g = row["100g_USD"]

        bounding_box = [
            (ord(loc_country[0]), ord(loc_country[0])),
            (rating, rating),
            (review_date, review_date),
            (usd_100g, usd_100g)
        ]
        rtree.insert(bounding_box, data=row.to_dict())


def query_rtree(rtree, loc_country=None, rating_range=None, review_date_range=None, usd_100g_range=None):
    loc_country_range = None
    if loc_country:
        loc_country_range = (ord(loc_country[0]), ord(loc_country[0]))

    query_bounds = [
        loc_country_range if loc_country_range else (-float("inf"), float("inf")),
        rating_range if rating_range else (-float("inf"), float("inf")),
        review_date_range if review_date_range else (-float("inf"), float("inf")),
        usd_100g_range if usd_100g_range else (-float("inf"), float("inf"))
    ]
    return rtree.search(query_bounds)


def create_minhash(text, num_perm=128):
    minhash = MinHash(num_perm=num_perm)
    for word in text.split():
        minhash.update(word.encode('utf8'))
    return minhash


def find_similar_reviews_with_lsh(results, n_similar=3):
    # Combine descriptions into a single string and create MinHash for each review
    for result in results:
        result.data['combined_desc'] = f"{result.data['desc_1']} {result.data['desc_2']} {result.data['desc_3']}"
        result.minhash = create_minhash(result.data['combined_desc'])

    # Create LSH index
    lsh = MinHashLSH(threshold=0.5, num_perm=128)
    for i, result in enumerate(results):
        lsh.insert(f"review_{i}", result.minhash)

    # Find similar reviews using LSH
    similar_reviews = {}
    for i, result in enumerate(results):
        similar = lsh.query(result.minhash)
        similar_reviews[i] = similar[:n_similar]

    return similar_reviews


if __name__ == "__main__":
    # Path to CSV file
    file_path = "C:/Users/tasis/Desktop/sxoli/code/quad-trees/coffee_analysis.csv"
    df = load_data(file_path)

    rtree = RTree(max_children=100, dimensions=4)
    insert_into_rtree(rtree, df)

    start_time = time.time()

    results = query_rtree(
        rtree,
        loc_country= 'United States',
        rating_range=(70, 99),
        review_date_range=(1, 1490),
        usd_100g_range=(1, 50)
    )

    # Find similar reviews using LSH
    similar_reviews = find_similar_reviews_with_lsh(results, n_similar=3)

    end_time = time.time()

    # Print the most similar reviews
    for i, similar in similar_reviews.items():
        print(f"Review {i+1} Similar Reviews:")
        for idx in similar:
            print(f"  Review Index: {idx}")
        print()

    for result in results:
        row = result.data
        print(
            f"100g_USD: {row['100g_USD']}, "
            f"loc_country: {row['loc_country']}, "
            f"review_date: {row['review_date']} (days since 01/01/2017), "
            f"rating: {row['rating']}"
        )

    print((end_time-start_time) * 1000 , " ms")