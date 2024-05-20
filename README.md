### TaxoGen implementation

You can create taxonomy by using **build_taxonomy** function from **taxogen.py**. It takes documents as list of list of words and terms as list of key terms and returns anytree root node class. You can acess node terms by acessing node's 'data' variable. It contains Topic class with list of terms and document id's, embeddings and scores dictionaries.

**main.ipynb** contains usage example.

Original paper:
"TaxoGen: Unsupervised Topic Taxonomy Construction by Adaptive Term Embedding and Clustering", Chao Zhang, Fangbo Tao, Xiusi Chen, Jiaming Shen, Meng Jiang, Brian Sadler, Michelle Vanni, Jiawei Han, ACM SIGKDD Conference on Knowledge Discovery and Pattern Mining (KDD), 2018.
