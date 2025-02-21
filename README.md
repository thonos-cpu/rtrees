🌲 R-Tree Spatial Indexing with MinHash & LSH 🌲
Welcome to the R-Tree Spatial Indexing project! This repository contains a Python implementation of R-Tree, a data structure that efficiently manages spatial data (such as geographical points, bounding boxes, etc.), along with MinHash and Locality Sensitive Hashing (LSH) for performing fast similarity search on reviews. This project is ideal for handling large datasets with spatial and textual information, enabling efficient spatial queries and similarity-based retrieval.

📂 Project Structure
This project consists of several key components:

RTree Class: Manages spatial data with efficient insertions, splits, and range queries.
MinHash and LSH: Used for fast similarity search of textual data (reviews) using Locality Sensitive Hashing.
Query Functions: Allows for user queries to find similar reviews and retrieve spatial data.
Data Processing: Loads and preprocesses the review dataset, extracting the relevant spatial attributes.
⚙️ Key Features
1. R-Tree Data Structure (Spatial Indexing)
The R-Tree is used to manage spatial data by organizing it into bounding boxes. This enables:

Efficient Insertions: The tree dynamically balances and splits as more points are inserted.
Bounding Box Querying: Perform spatial queries based on ranges for various dimensions like rating, review date, and price.
Fast Search: Queries find points that intersect with given ranges (bounding boxes).
2. MinHash & LSH for Similarity Search
MinHash is used to hash textual data (reviews) into compact signatures that preserve similarity.
LSH (Locality Sensitive Hashing) enables fast similarity searches, making it possible to find similar reviews quickly based on combined descriptions.
🚀 Getting Started
🛠 Installation
To run the project locally, follow these steps:

Clone the repository to your machine:

bash
git clone https://github.com/thonos-cpu/rtrees.git
cd rtrees
Install the necessary dependencies:

bash
pip install -r requirements.txt
The requirements.txt file includes dependencies like:

pandas for data handling
datasketch for MinHash and LSH functionality
sklearn for Nearest Neighbor search (optional)
datetime and time for processing timestamps
⚡ Run the Code
To run the program:

Ensure that your CSV data (coffee_analysis.csv) is available in the correct location.
Execute the script:
bash
python rtree_minhash_lsh.py

📄 File Breakdown
📁 RTree Class - Efficient Spatial Indexing
Node Class: Represents a node in the R-Tree. Nodes can be leaves or internal nodes, and each node contains bounding boxes and child nodes.
RTree Class: Handles insertions of spatial data (bounding boxes), splits nodes when the capacity exceeds the max children, and supports range queries.
Insert: Add a new node with a given bounding box and associated data.
Search: Search for nodes whose bounding boxes intersect with a given query range.
Split: When a node exceeds its capacity, it splits into two nodes to maintain balanced tree structure.

📁 MinHash & LSH - Similarity Search for Reviews
create_minhash Function: Creates a MinHash signature for a given review text. This signature can be used to perform approximate similarity searches.
find_similar_reviews_with_lsh Function: Uses LSH to find similar reviews based on the MinHash signatures of their combined descriptions.

📁 Data Loading and Querying - Handling CSV Data and User Queries
load_data Function: Loads a CSV file containing review data and processes it into numerical formats.
Converts the review_date into days since January 1, 2017.
insert_into_rtree Function: Inserts the processed data into the R-Tree.
query_rtree Function: Performs a spatial query on the R-Tree based on user-defined criteria (e.g., location, rating, price).

🧠 How It Works
Data Loading: The CSV file (e.g., coffee_analysis.csv) is loaded, and the reviews are preprocessed.

The review_date is converted to the number of days since January 1, 2017.
A bounding box is generated for each review using rating, review_date, price, and location.
R-Tree Construction: The R-Tree is populated with the bounding boxes of reviews. The tree handles spatial insertions, balancing, and querying.

Similarity Search with MinHash & LSH:

MinHash is used to generate signatures from the combined descriptions of reviews.
LSH is then used to find similar reviews based on these signatures.
User Queries: The user can perform a spatial query to find reviews that match specific location, rating, review date, and price ranges. Similar reviews are then retrieved using LSH.

Example Query
python
# Query for reviews from the United States with a rating between 70 and 99, price between 1 and 50 USD, and review date between 1 and 1490 days.
results = query_rtree(
    rtree,
    loc_country='United States',
    rating_range=(70, 99),
    review_date_range=(1, 1490),
    usd_100g_range=(1, 50)
)
Find Similar Reviews
python
# Use LSH to find similar reviews to the ones in the query results
similar_reviews = find_similar_reviews_with_lsh(results, n_similar=3)
📊 Data Visualization and Output
The results from the spatial query and the similar reviews found via LSH are printed to the console.

Sample output for a query:

plaintext
Review 1 Similar Reviews:
  Review Index: 3
  Review Index: 5

Review 2 Similar Reviews:
  Review Index: 2
  Review Index: 4

...

100g_USD: 12.5, loc_country: United States, review_date: 150, rating: 80
🔧 Performance
The project measures the execution time for query searches and similarity searches. The time is printed in milliseconds for performance analysis:

python
print((end_time - start_time) * 1000, "ms")
🚀 Contributing
We welcome contributions! To get involved:

Fork the repository.
Clone your fork locally.
Create a new branch for your changes.
Push your changes and submit a pull request.
