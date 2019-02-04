#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:10:35 2018

@author: kathleenyoung
Kathleen Young
IEMS 308 Assignment 2: Association Rules
Copyright 2018

Homework 2
"""

#Importing necessary packages
import pandas as pd
import psycopg2
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#Connect to postegres database
conn = None
try:
    conn = psycopg2.connect(
            "host='gallery.iems.northwestern.edu' dbname='iems308'user = '*****' password = '*****'")
except:
    print("Can't connect! :/")
    
#################
#Data exploration
#################

#Decoding the pos.trnsact table
#c1: sku
#c2: store
#c3: register
#c4: trannum (transaction code)
#c5: seq (sequence number)?
#c6: sale date
#c7: stype (return or purchase)
#c8: quantity (item quantity of the transaction)
#c9: amt (total amount of the transaction charge to the customer)?
#c10: ? orgprice (original price of the item stock)
#c11: ? orgprice (original price of the item stock)  
#c12: interid (internal id)?
#c13: mic
#c14: always zero?

#Number of enteries in trnsact table
num_trnsact = pd.read_sql(
        "SELECT COUNT(*) FROM pos.trnsact",
        con=conn)

#Number of stores
num_stores = pd.read_sql(
        "SELECT COUNT(*) FROM pos.strinfo",
        con=conn)
print(num_stores)

#List of stores
stores = pd.read_sql(
        "SELECT * FROM pos.strinfo",
        con=conn)

#Number of SKUs
num_sku = pd.read_sql("SELECT COUNT(*) FROM pos.skuinfo",
                      con=conn)
print(num_sku)

#List of SKUs
skus = pd.read_sql("SELECT * FROM pos.skuinfo",
                   con=conn)

#Number of departments
num_depts = pd.read_sql("SELECT COUNT(*) FROM pos.deptinfo",
                        con=conn)
print(num_depts)

#Departments table
depts = pd.read_sql("SELECT * FROM pos.deptinfo",
                    con=conn)

#Number SKUs in departmetn 800 (CLINIQUE)
num_skus_800 = pd.read_sql("SELECT COUNT(*) FROM pos.skuinfo WHERE pos.skuinfo.dept = 800",
                       con=conn)
print(num_skus_800)

#SKUs in department 800 (CLINIQUE)
dept_800 = pd.read_sql("SELECT * FROM pos.skuinfo WHERE pos.skuinfo.dept = 800",
                       con=conn)

#Number SKUs in departmetn 1704 (RALPH LAUREN)
num_skus_1704 = pd.read_sql("SELECT COUNT(*) FROM pos.skuinfo WHERE pos.skuinfo.dept = 1704",
                       con=conn)
print(num_skus_1704)

#SKUs in department 1704 (RALPH LAUREN)
dept_1704 = pd.read_sql("SELECT * FROM pos.skuinfo WHERE pos.skuinfo.dept = 1704",
                       con=conn)

#Number SKUs in departmetn 9306 (SPERRY)
num_skus_9306 = pd.read_sql("SELECT COUNT(*) FROM pos.skuinfo WHERE pos.skuinfo.dept = 9306",
                       con=conn)
print(num_skus_9306)

#SKUs in department 9306 (SPERRY)
dept_9306 = pd.read_sql("SELECT * FROM pos.skuinfo WHERE pos.skuinfo.dept = 9306",
                       con=conn)

################
#Creating tables
################

#This was generally done in postgreSQL

#Create a table with all transactions from 2005-05-07
#pd.read_sql("CREATE TABLE kay498_schema.date AS SELECT * FROM pos.trnsact WHERE pos.trnsact.c6 = '2005-05-07'",
#                      con=conn)

#Create a new table with SKUs as strings so it can be INNER JOINed with pos.trnsact
#pd.read_sql("CREATE TABLE kay498_schema.new_table AS SELECT pos.skuinfo.sku, pos.skuinfo.dept FROM pos.skuinfo",
#                        con=conn)

#Create a table that INNER JOINs the new_table and pos.trnsact on SKUs
#(pos.trnsact now includes dept info)
#This did not work because I did not have enough space in my schema
#pd.read_sql("CREATE TABLE kay498_schema.trn_dept AS SELECT * FROM pos.trnsact INNER JOIN kay498_schema.new_table ON (pos.trnsact.c1 = kay498_schema.new_table.sku)",
#            con=conn)

#Create a table with the moline and department data
#CREATE TABLE kay498_schema.trnsact_dept AS
#SELECT * FROM kay498_schema.moline
#INNER JOIN kay498_schema.new_table ON (kay498_schema.moline.c1 = kay498_schema.new_table.sku)

#Moline dataframe
df_moline = pd.read_sql("SELECT * FROM kay498_schema.moline",
                        con=conn)
#Moline Clinique dataframe
df_moline_clinique = pd.read_sql("SELECT * FROM kay498_schema.trnsact_dept WHERE kay498_schema.trnsact_dept.dept = 800",
                                 con = conn)
df_m_c_count = pd.read_sql("SELECT COUNT(*) FROM kay498_schema.trnsact_dept WHERE kay498_schema.trnsact_dept.dept = 800",
                                 con = conn)
print(df_m_c_count)

#Moline Ralph Lauren dataframe
df_moline_rlauren = pd.read_sql("SELECT * FROM kay498_schema.trnsact_dept WHERE kay498_schema.trnsact_dept.dept = 1704",
                                 con = conn)
df_m_rl = pd.read_sql("SELECT COUNT(*) FROM kay498_schema.trnsact_dept WHERE kay498_schema.trnsact_dept.dept = 1704",
                                 con = conn)
print(df_m_rl)

#Moline dept 1100 dataframe
df_moline_1100 = pd.read_sql("SELECT * FROM kay498_schema.trnsact_dept WHERE kay498_schema.trnsact_dept.dept = 1100",
                                 con = conn)

###########################
#Creating Association Rules
###########################

#For a single dataframe
#Pick the dataframe
df = df_moline_1100

#Force the quantity column to be numeric (for one-hot)
df[['c8']] = df[['c8']].apply(pd.to_numeric)

#Create an "index" by combining all of the primary key columns into a single
#column.
df["index"] = df["c2"] + df["c3"]+ df["c4"]+ df["c5"]

#One-hot encode the whole damn thing
basket = (df.groupby(['index', 'c1'])['c8'].sum().unstack().reset_index().
          fillna(0).set_index('index'))

#Force the quantity column to only be between 0 and 1 (for one-hot)
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
basket_sets = basket.applymap(encode_units)

#Find frequent_itemsets with support of at least 5%
frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)

#Make some rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)


#Looping through departments

#Setting up variables
depts_list = depts.dept
df_dict = dict()
basket_dict = dict()
f_itemsets_dict = dict()
rules_dict = dict()
basket_sets_dict = dict()

#Making dataframes for each department
for dept in depts_list:
    df_dict[dept] = pd.read_sql(f"SELECT * FROM kay498_schema.trnsact_dept WHERE kay498_schema.trnsact_dept.dept = '{dept}'",
       con = conn)

#Making sure everything is numeric
for dept in depts_list:
    df_dict[dept][['c8']] = df_dict[dept][['c8']].apply(pd.to_numeric)

#Create an index for each new dataframe
for dept in depts_list:
    df_dict[dept]["index"] = df_dict[dept]["c2"] + df_dict[dept]["c3"]+ df_dict[dept]["c4"]+ df_dict[dept]["c5"]

#One-hot encode everything   
for dept in depts_list:
    if len(df_dict[dept].c1) > 0:
        basket_dict[dept] = (df_dict[dept].groupby(['index', 'c1'])['c8'].sum().unstack().
                   reset_index().fillna(0).set_index('index'))

#Force the one-hot values to be either 0 or 1
for dept in depts_list:
    if dept != 4407 and dept != 5506 and dept != 8002:
        basket_sets_dict[dept] = basket_dict[dept].applymap(encode_units)

#Find the frequent item sets, except the ones that don't work
for dept in depts_list:
    if dept != 4407 and dept != 5506 and dept != 8002:
        f_itemsets_dict[dept] = apriori(basket_sets_dict[dept], min_support=0.05, use_colnames=True)

#A list of the specific departments with lots of frequent item sets
high_sup = [1100, 3100, 4400, 6400, 7200, 7205, 8000, 9000]

#Loop through these and make rules for them
for dept in high_sup:
    rules_dict[dept] = association_rules(f_itemsets_dict[dept], metric="lift", min_threshold=1)