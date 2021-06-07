from collections import defaultdict

import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import fpgrowth, association_rules, apriori
from mlxtend.preprocessing import TransactionEncoder

df = pd.read_csv(r"cleaned_data.csv", usecols=["User_Id", "Category_Id", "Behavior_type"])
# print(df.columns)
# ['User_Id', 'Item_Id', 'Category_Id', 'Behavior_type', 'Timestamp','Date', 'Time']
# print(df["Behavior_type"].value_counts())
# pv  cart  fav  buy

df = df.loc[df["Behavior_type"] == "buy"]
df = df.drop_duplicates(['User_Id', 'Category_Id']).copy()
df_list = df.groupby('User_Id')['Category_Id'].apply(list)
df_list = list(df_list)

te = TransactionEncoder()
te_ary = te.fit(df_list).transform(df_list)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = fpgrowth(df, min_support=0.001, use_colnames=True).sort_values('support', ascending=False)
print(frequent_itemsets)
frequent_itemsets.to_csv('Category_fi.csv', index=0) #不保存行索引

# 获取置信度>=0.1的关联规则，并按提升度倒序排列
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1).sort_values('support', ascending=False)

print(rules)
rules.to_csv('Category_ru.csv', index=0) #不保存行索引

cate1 = list(rules.loc[rules["support"] > 0.002]["antecedents"])
cate2 = list(rules.loc[rules["support"] > 0.002]["consequents"])
tmp = (cate1 + cate2)
top_category_id = list(set([list(x)[0] for x in tmp]))
print(top_category_id)
# ============================================================================================
# top support category
# 0.045309745	frozenset({1464116})
# 0.042709398	frozenset({2735466})
# 0.041798166	frozenset({4145813})
# 0.039775674	frozenset({2885642})
# 0.034878725	frozenset({4756105})
# 0.033619297	frozenset({4801426})
# 0.03148568	frozenset({982926})


df = pd.read_csv(r"cleaned_data.csv", usecols=["User_Id", "Item_Id", "Category_Id", "Behavior_type"])
df = df.loc[df["Behavior_type"] == "buy"]

df = df[df['Category_Id'].isin([1464116, 2735466, 4145813, 2885642, 4756105, 4801426, 982926])] #筛选出需要分析的商品种类

df = df.drop_duplicates(['User_Id', 'Item_Id']).copy()
df_list = df.groupby('User_Id')['Item_Id'].apply(list)
df_list = list(df_list)
te = TransactionEncoder()
te_ary = te.fit(df_list).transform(df_list)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = fpgrowth(df, min_support=0.001, use_colnames=True).sort_values('support', ascending=False)
print(frequent_itemsets)
frequent_itemsets.to_csv('item_fi.csv', index=0) #不保存行索引

# top support category
# frozenset({4159072})	frozenset({2885642})
# frozenset({2885642})	frozenset({4159072})
# frozenset({4801426})	frozenset({982926})
# frozenset({982926})	frozenset({4801426})
# frozenset({982926})	frozenset({2735466})
# frozenset({1320293})	frozenset({982926})
# frozenset({1320293})	frozenset({4801426})
# frozenset({2465336})	frozenset({4801426})
# frozenset({1320293})	frozenset({2735466})
# frozenset({3738615})	frozenset({2640118})
# frozenset({4217906})	frozenset({4756105})
# frozenset({4789432})	frozenset({1464116})
# frozenset({149192})	frozenset({982926})



df = pd.read_csv(r"cleaned_data.csv", usecols=["User_Id", "Item_Id", "Category_Id", "Behavior_type"])
df = df.loc[df["Behavior_type"] == "buy"]

df = df[df['Category_Id'].isin([4159072, 1320293, 149192, 4756105, 2885642, 2735466, 982926, 4801426, 4217906, 1464116, 2640118, 3738615, 2465336, 4789432]

)] #筛选出需要分析的商品种类

df = df.drop_duplicates(['User_Id', 'Item_Id']).copy()
df_list = df.groupby('User_Id')['Item_Id'].apply(list)
df_list = list(df_list)
te = TransactionEncoder()
te_ary = te.fit(df_list).transform(df_list)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = fpgrowth(df, min_support=0.0001, use_colnames=True).sort_values('support', ascending=False)
# 获取置信度>=0.1的关联规则，并按提升度倒序排列
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.0001).sort_values('support', ascending=False)
rules.to_csv('item_ru.csv', index=0) #不保存行索引
print(rules)


def find_frequent_itemsets(transactions, minimum_support, include_support=False):

    items = defaultdict(lambda: 0)  # mapping from items to their supports

    # Load the passed-in transactions and count the support that individual
    # items have.
    for transaction in transactions:
        for item in transaction:
            items[item] += 1

    # Remove infrequent items from the item support dictionary.
    items = dict((item, support) for item, support in items.items()
        if support >= minimum_support)

    # Build our FP-tree. Before any transactions can be added to the tree, they
    # must be stripped of infrequent items and their surviving items must be
    # sorted in decreasing order of frequency.
    def clean_transaction(transaction):
        transaction = filter(lambda v: v in items, transaction)
        transaction_list = list(transaction)   # 为了防止变量在其他部分调用，这里引入临时变量transaction_list
        transaction_list.sort(key=lambda v: items[v], reverse=True)
        return transaction_list

