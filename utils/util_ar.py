# general imports
import pandas as pd
from skimpy import skim
import ast
import seaborn as sns
import matplotlib.pyplot as plt


# Mlxtend library
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder



def create_ar(cluster, metric:str = "confidence", support:float = 0.05, min_threshold:float = 0.2):
    
    list_of_goods = [ast.literal_eval(product) for product in cluster["list_of_goods"].values]
    
    te = TransactionEncoder()
    te_fit = te.fit(list_of_goods).transform(list_of_goods)
    cluster_items = pd.DataFrame(te_fit, columns=te.columns_)
    
    frequent_cluster = apriori(
    cluster_items, min_support=support, use_colnames=True
    )

    frequent_cluster.sort_values(by='support', ascending=False)
    
    rules_cluster = association_rules(frequent_cluster, 
                                  metric=metric, 
                                  min_threshold=min_threshold)
    rules_cluster = rules_cluster[rules_cluster["lift"]>=1.05].sort_values(by='support', ascending=False)
    return rules_cluster


def education_pie(cluster):
    sns.set_style("white")

    plt.subplots(figsize=(5, 5))

    labels = "Basic","Baschelor","Master", "Phd"
    size = 0.5

    colors = ["#682F2F", "#F3AB60", "#9c754e", "#e38019"]  # Unique colors for each education level


    wedges, texts, autotexts = plt.pie([cluster["education"].value_counts()[0],
                                        cluster["education"].value_counts()[1],
                                        cluster["education"].value_counts()[2],
                                        cluster["education"].value_counts()[3]],
                                        explode = (0,0,0,0),
                                        textprops=dict(size= 15, color= "white"),
                                        autopct="%.2f%%", 
                                        pctdistance = 0.72,
                                        radius=.9, 
                                        colors = ["#682F2F","#F3AB60","#9F8A78","#4f4842"], 
                                        shadow = True,
                                        wedgeprops=dict(width = size, edgecolor = "black", 
                                        linewidth = 1),
                                        startangle = 20)

    plt.legend(wedges, labels, title="Category",loc="center left",bbox_to_anchor=(1, 0, 0.5, 1), edgecolor = "black")
    plt.title("\nCustomer's Education Level",fontsize=20)
    plt.show()


def gender_pie(cluster):
    sns.set_style("white")

    plt.subplots(figsize=(5, 5))

    labels = "Female","Male"
    size = 0.5

    colors = ["#ff6f9c", "#4169e1"]


    wedges, texts, autotexts = plt.pie([cluster["customer_gender"].value_counts()[0],
                                        cluster["customer_gender"].value_counts()[1]],
                                        explode = (0,0),
                                        textprops=dict(size= 15, color= "white"),
                                        autopct="%.2f%%", 
                                        pctdistance = 0.72,
                                        radius=.9, 
                                        colors = ["#ff6f9c", "#4169e1"], 
                                        shadow = True,
                                        wedgeprops=dict(width = size, edgecolor = "black", 
                                        linewidth = 1),
                                        startangle = 20)

    plt.legend(wedges, labels, title="Category",loc="center left",bbox_to_anchor=(1, 0, 0.5, 1), edgecolor = "black")
    plt.title("\nCustomer's Gender",fontsize=20)
    plt.show()


def loyalty_card_pie(cluster):
    sns.set_style("white")

    plt.subplots(figsize=(5, 5))

    labels = "Has loyalty card","Does not have loyalty card"
    size = 0.5

    colors = ["#24b89f", "#db4760"]


    wedges, texts, autotexts = plt.pie([cluster["has_loyalty_card"].value_counts()[0],
                                        cluster["has_loyalty_card"].value_counts()[1]],
                                        explode = (0,0),
                                        textprops=dict(size= 15, color= "white"),
                                        autopct="%.2f%%", 
                                        pctdistance = 0.72,
                                        radius=.9, 
                                        colors = ["#24b89f", "#db4760"], 
                                        shadow = True,
                                        wedgeprops=dict(width = size, edgecolor = "black", 
                                        linewidth = 1),
                                        startangle = 20)

    plt.legend(wedges, labels, title="Category",loc="center left",bbox_to_anchor=(1, 0, 0.5, 1), edgecolor = "black")
    plt.title("\nCustomer's loyalty card ownership",fontsize=20)
    plt.show()