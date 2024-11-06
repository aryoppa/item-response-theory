from itertools import combinations
from collections import defaultdict
import pandas as pd

class Apriori:
    """
    This class is for handling apriori algorithm
    """

    def generate_single_items(self, transactions):
        C1 = []
        for transaction in transactions:
            for item in transaction:
                if [item] not in C1:
                    C1.append([item])
        C1.sort()
        return C1

    def prune(self, candidates, transactions, min_support):
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
        Ck_plus_1 = []
        len_Lk = len(Lk)
        for i in range(len_Lk):
            for j in range(i + 1, len_Lk):
                l1, l2 = Lk[i], Lk[j]
                if l1[:-1] == l2[:-1] and l1[-1] != l2[-1]:
                    candidate = sorted(list(set(l1) | set(l2)))
                    if candidate not in Ck_plus_1:
                        Ck_plus_1.append(candidate)
        return Ck_plus_1

    def apriori(self, transactions, min_support):
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