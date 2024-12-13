from collections import defaultdict

class TreeNode:
    """
    This class represents a node in the FP-Tree.
    """
    def __init__(self, name, count, parent):
        self.name = name
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None

class FpGrowth:
    """
    Class for implementing the FP-Growth algorithm.
    """

    def build_tree(self, dataset, min_support):
        """
        Builds the FP-Tree from the given dataset.
        """
        header_table = defaultdict(int)
        for transaction, count in dataset.items():
            for item in transaction:
                header_table[item] += count

        header_table = {
            k: [v, None] for k, v in header_table.items() if v >= min_support
        }
        
        if not header_table:
            return None, None

        fp_tree = TreeNode('Null', 1, None)

        for transaction, count in dataset.items():
            filtered_transaction = [
                item for item in transaction if item in header_table
            ]
            filtered_transaction.sort(
                key=lambda x: header_table[x][0], reverse=True
            )
            if filtered_transaction:
                self.update_tree(filtered_transaction, fp_tree, header_table, count)

        return fp_tree, header_table

    def update_tree(self, transaction, node, header_table, count):
        """
        Adds a transaction to the FP-Tree.
        """
        first_item = transaction[0]
        if first_item in node.children:
            node.children[first_item].count += count
        else:
            child_node = TreeNode(first_item, count, node)
            node.children[first_item] = child_node
            if header_table[first_item][1] is None:
                header_table[first_item][1] = child_node
            else:
                self.update_header(header_table[first_item][1], child_node)

        if len(transaction) > 1:
            self.update_tree(transaction[1:], node.children[first_item], header_table, count)

    def update_header(self, node, target_node):
        """
        Updates the header table with a link to the next node of the same item.
        """
        while node.link is not None:
            node = node.link
        node.link = target_node

    def projecting_tree(self, item, header_table):
        """
        Generates conditional pattern bases for a given item.
        """
        conditional_pattern_bases = defaultdict(int)
        node = header_table[item][1]
        while node:
            prefix_path = []
            current_node = node.parent
            while current_node and current_node.name != 'Null':
                prefix_path.append(current_node.name)
                current_node = current_node.parent
            if prefix_path:
                conditional_pattern_bases[frozenset(prefix_path)] += node.count
            node = node.link
        return conditional_pattern_bases

    def fp_growth(self, dataset, min_support):
        """
        Main FP-Growth algorithm to find frequent patterns.
        """
        fp_tree, header_table = self.build_tree(dataset, min_support)
        if not fp_tree:
            return []

        frequent_patterns = []
        for item in sorted(header_table.keys(), key=lambda x: header_table[x][0]):
            conditional_pattern_bases = self.projecting_tree(item, header_table)
            conditional_fp_tree, _ = self.build_tree(conditional_pattern_bases, min_support)
            if conditional_fp_tree:
                sub_patterns = self.fp_growth(conditional_pattern_bases, min_support)
                for pattern in sub_patterns:
                    frequent_patterns.append(pattern.union({item}))
            else:
                frequent_patterns.append({item})

        return frequent_patterns
