from itertools import combinations
from collections import defaultdict

class Apriori:
    """
    This class implements the Apriori algorithm for frequent itemset mining.

    Methods:
        generate_single_items(transactions):
            Generates unique single-item candidates from the transactions.

        prune(candidates, transactions, min_support):
            Prunes candidate itemsets based on minimum support threshold.

        generate_candidates(Lk):
            Generates k+1-item candidates from k-item frequent itemsets.

        apriori(transactions, min_support):
            Runs the Apriori algorithm and returns frequent itemsets with their support values.
    """

    def generate_single_items(self, transactions):
        """
        Generate unique single-item candidates from the transactions.

        Args:
            transactions (list of list): The dataset of transactions.

        Returns:
            list: A list of unique single-item candidates.
        """
        C1 = []
        for transaction in transactions:
            for item in transaction:
                if [item] not in C1:
                    C1.append([item])
        C1.sort()
        return C1

    def prune(self, candidates, transactions, min_support):
        """
        Prunes candidate itemsets by removing those with support less than the minimum threshold.

        Args:
            candidates (list of list): Candidate itemsets.
            transactions (list of list): The dataset of transactions.
            min_support (float): Minimum support threshold.

        Returns:
            tuple: A list of frequent itemsets and a dictionary of their support values.
        """
        Lk = []
        item_count = defaultdict(int)
        num_transactions = len(transactions)

        for transaction in transactions:
            for candidate in candidates:
                if set(candidate).issubset(set(transaction)):
                    item_count[tuple(candidate)] += 1

        support_data = {}
        for candidate in candidates:
            support = item_count[tuple(candidate)] / num_transactions
            if support >= min_support:
                Lk.append(candidate)
                support_data[tuple(candidate)] = support

        return Lk, support_data

    def generate_candidates(self, Lk):
        """
        Generate k+1-item candidate itemsets from k-item frequent itemsets.

        Args:
            Lk (list of list): k-item frequent itemsets.

        Returns:
            list: A list of k+1-item candidate itemsets.
        """
        Ck_plus_1 = []
        len_Lk = len(Lk)
        for i in range(len_Lk):
            for j in range(i + 1, len_Lk):
                l1, l2 = Lk[i], Lk[j]
                if l1[:-1] == l2[:-1] and l1[-1] != l2[-1]:
                    candidate = sorted(set(l1) | set(l2))
                    if candidate not in Ck_plus_1:
                        Ck_plus_1.append(candidate)
        return Ck_plus_1

    def apriori(self, transactions, min_support):
        """
        Run the Apriori algorithm to find frequent itemsets.

        Args:
            transactions (list of list): The dataset of transactions.
            min_support (float): Minimum support threshold.

        Returns:
            dict: Frequent itemsets with their support values.
        """
        C1 = self.generate_single_items(transactions)
        L1, support_data = self.prune(C1, transactions, min_support)
        L = [L1]
        k = 1
        
        while L[k-1]:
            Ck_plus_1 = self.generate_candidates(L[k-1])
            Lk_plus_1, sup_data = self.prune(Ck_plus_1, transactions, min_support)
            support_data.update(sup_data)
            L.append(Lk_plus_1)
            k += 1

        return support_data