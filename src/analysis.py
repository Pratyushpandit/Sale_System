"""
Diwali Sales Data Analysis
Full coursework script - run top to bottom
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import skew, kurtosis, f_oneway, chi2_contingency
from pandas.api.types import CategoricalDtype
import warnings
warnings.filterwarnings('ignore')

# Paths are resolved relative to the project root (parent of this file's
# `src/` folder) so the script can be run from any working directory.
ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / 'data' / 'Diwali Sales Data.csv'
OUT = str(ROOT / 'outputs') + '/'
Path(OUT).mkdir(parents=True, exist_ok=True)

# ============================================================
# SECTION 1 - DATA UNDERSTANDING
# ============================================================
print("=" * 60)
print("SECTION 1: DATA UNDERSTANDING")
print("=" * 60)

df_raw = pd.read_csv(DATA_FILE, encoding='latin-1')
print("\nShape:", df_raw.shape)
print("\nColumn names:\n", df_raw.columns.tolist())
print("\nData types:\n", df_raw.dtypes)
print("\nMissing values per column:\n", df_raw.isnull().sum())
print("\nFirst 5 rows:")
print(df_raw.head())


# ============================================================
# SECTION 2 - DATA PREPARATION
# ============================================================
print("\n" + "=" * 60)
print("SECTION 2: DATA PREPARATION")
print("=" * 60)

df = pd.read_csv(DATA_FILE, encoding='latin-1')
print("\nDataset loaded successfully")
print("Shape:", df.shape)
print("\nBasic info:")
print(df.info())

print("\nBusiness column meanings:")
meanings = {
    'User_ID':          'Unique customer ID',
    'Cust_name':        'Customer name (not useful for analysis)',
    'Product_ID':       'Product identifier',
    'Gender':           'Customer gender M or F',
    'Age Group':        'Age range as text e.g. 26-35',
    'Age':              'Numeric age',
    'Marital_Status':   '0 = unmarried, 1 = married',
    'State':            'State where customer lives',
    'Zone':             'Region (Central, Western, Southern, Northern)',
    'Occupation':       'Customer job type',
    'Product_Category': 'Type of product purchased',
    'Orders':           'Number of orders placed',
    'Amount':           'Total purchase amount in rupees (main target)',
    'Status':           'Mostly empty column',
    'unnamed1':         'Completely empty column'
}
for col, meaning in meanings.items():
    print(f"  {col:20s}: {meaning}")


# ============================================================
# SECTION 3 - DATA CLEANING
# ============================================================
print("\n" + "=" * 60)
print("SECTION 3: DATA CLEANING")
print("=" * 60)

# Converting Age Group to ordinal categorical
print("\n--- 3.1 Age Group to ordinal category ---")
age_map = {
    '0-17':  'Teen',
    '18-25': 'Young Adult',
    '26-35': 'Adult',
    '36-45': 'Middle Aged',
    '46-50': 'Senior',
    '51-55': 'Senior',
    '55+':   'Senior'
}
df['Age_Group_Label'] = df['Age Group'].map(age_map)
age_order = ['Teen', 'Young Adult', 'Adult', 'Middle Aged', 'Senior']
age_cat = CategoricalDtype(categories=age_order, ordered=True)
df['Age_Group_Label'] = df['Age_Group_Label'].astype(age_cat)
print("Unique Age Group Labels:", df['Age_Group_Label'].unique().tolist())
print(df['Age_Group_Label'].value_counts())

# Creating Purchase_Value_Category
print("\n--- 3.2 Purchase Value Category ---")
df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
low_cut  = df['Amount'].quantile(0.33)
high_cut = df['Amount'].quantile(0.66)
print(f"Low cutoff (<= {low_cut:.0f}), High cutoff (> {high_cut:.0f})")

def label_amount(val):
    if pd.isna(val):
        return np.nan
    elif val <= low_cut:
        return 'Low'
    elif val <= high_cut:
        return 'Medium'
    else:
        return 'High'

df['Purchase_Value_Category'] = df['Amount'].apply(label_amount)
print(df['Purchase_Value_Category'].value_counts())

# Dropping irrelevant columns
print("\n--- 3.3 Drop irrelevant columns ---")
columns_to_drop = ['Cust_name', 'User_ID', 'unnamed1']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
print("Columns after drop:", df.columns.tolist())

# Handling missing values
print("\n--- 3.4 Handle missing values ---")
print("Missing before:")
print(df[['Amount', 'Age']].isnull().sum())
df.dropna(subset=['Amount', 'Age'], inplace=True)
print("Missing after:")
print(df[['Amount', 'Age']].isnull().sum())
print("Rows remaining:", len(df))

# Unique values for categorical columns
print("\n--- 3.5 Unique values for categorical columns ---")
cat_cols = ['Product_Category', 'Zone', 'Gender', 'Occupation', 'Marital_Status']
for col in cat_cols:
    print(f"\n{col} ({df[col].nunique()} unique):")
    print(" ", df[col].unique().tolist())


# ============================================================
# SECTION 4 - DATA ANALYSIS
# ============================================================
print("\n" + "=" * 60)
print("SECTION 4: DATA ANALYSIS")
print("=" * 60)

# Summary statistics
print("\n--- 4.1 Summary Statistics ---")
num_cols = ['Age', 'Orders', 'Amount']
stats = df[num_cols].describe()
stats.loc['skewness'] = df[num_cols].apply(skew)
stats.loc['kurtosis'] = df[num_cols].apply(kurtosis)
print(stats.round(4))

# Correlation
print("\n--- 4.2 Correlation Analysis ---")
corr = df[['Age', 'Orders', 'Amount']].corr()
print("\nCorrelation matrix:")
print(corr.round(4))
print(f"\nAge vs Amount:    {df['Age'].corr(df['Amount']):.4f}")
print(f"Orders vs Amount: {df['Orders'].corr(df['Amount']):.4f}")


# ============================================================
# SECTION 5 - DATA EXPLORATION (VISUALIZATIONS)
# ============================================================
print("\n" + "=" * 60)
print("SECTION 5: DATA EXPLORATION")
print("=" * 60)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120

# Plot 1 - Top 5 states by total Amount
print("\nGenerating Plot 1: Top 5 States by Total Amount...")
top_states = df.groupby('State')['Amount'].sum().sort_values(ascending=False).head(5)
fig, ax = plt.subplots(figsize=(8, 5))
top_states.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
ax.set_title('Top 5 States by Total Sales Amount', fontsize=14)
ax.set_ylabel('Total Amount (Rupees)')
ax.set_xlabel('State')
ax.tick_params(axis='x', rotation=30)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50000,
            f'{bar.get_height()/1e6:.1f}M', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(OUT + 'plot1_top5_states.png')
plt.close()
print("  Saved: plot1_top5_states.png")
print("  Top 5 states:\n ", top_states.round(0).to_dict())

# Plot 2 - Gender distribution across Product_Category
print("\nGenerating Plot 2: Gender across Product Category...")
gender_cat = df.groupby(['Product_Category', 'Gender']).size().unstack(fill_value=0)
fig, ax = plt.subplots(figsize=(12, 6))
gender_cat.plot(kind='bar', ax=ax, color=['#FF69B4','#4169E1'], edgecolor='black')
ax.set_title('Gender Distribution Across Product Categories', fontsize=14)
ax.set_ylabel('Number of Customers')
ax.set_xlabel('Product Category')
ax.tick_params(axis='x', rotation=40)
ax.legend(title='Gender')
plt.tight_layout()
plt.savefig(OUT + 'plot2_gender_product.png')
plt.close()
print("  Saved: plot2_gender_product.png")

# Plot 3 - Age Group vs Average Amount
print("\nGenerating Plot 3: Age Group vs Average Amount...")
avg_amt = df.groupby('Age_Group_Label', observed=True)['Amount'].mean()
fig, ax = plt.subplots(figsize=(8, 5))
avg_amt.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
ax.set_title('Age Group vs Average Purchase Amount', fontsize=14)
ax.set_ylabel('Average Amount (Rupees)')
ax.set_xlabel('Age Group')
ax.tick_params(axis='x', rotation=25)
for bar in ax.patches:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(OUT + 'plot3_agegroup_amount.png')
plt.close()
print("  Saved: plot3_agegroup_amount.png")
print("  Avg amount by age group:\n ", avg_amt.round(0).to_dict())

# Plot 4 - Marital Status impact on Orders
print("\nGenerating Plot 4: Marital Status vs Orders...")
marital_orders = df.groupby('Marital_Status')['Orders'].mean()
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(['Unmarried (0)', 'Married (1)'], marital_orders.values,
              color=['#FFA500','#32CD32'], edgecolor='black')
ax.set_title('Marital Status Impact on Average Orders', fontsize=13)
ax.set_ylabel('Average Orders')
ax.set_xlabel('Marital Status')
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.savefig(OUT + 'plot4_marital_orders.png')
plt.close()
print("  Saved: plot4_marital_orders.png")
print("  Avg orders by marital status:\n ", marital_orders.round(3).to_dict())

# Plot 5 - Grouped bar: Product Category by Zone
print("\nGenerating Plot 5: Product Category ranked by Amount grouped by Zone...")
zone_cat = df.groupby(['Zone', 'Product_Category'])['Amount'].mean().reset_index()
zone_cat['Rank'] = zone_cat.groupby('Zone')['Amount'].rank(ascending=False).astype(int)
pivot = zone_cat.pivot(index='Product_Category', columns='Zone', values='Amount')
fig, ax = plt.subplots(figsize=(14, 6))
pivot.plot(kind='bar', ax=ax, edgecolor='black')
ax.set_title('Average Amount by Product Category and Zone', fontsize=14)
ax.set_ylabel('Average Amount (Rupees)')
ax.set_xlabel('Product Category')
ax.tick_params(axis='x', rotation=40)
ax.legend(title='Zone', loc='upper right')
plt.tight_layout()
plt.savefig(OUT + 'plot5_zone_category.png')
plt.close()
print("  Saved: plot5_zone_category.png")
print("\n  Ranked table (Zone + Product Category):")
print(zone_cat.sort_values(['Zone', 'Rank']).to_string(index=False))


# ============================================================
# SECTION 6 - STATISTICAL TESTING
# ============================================================
print("\n" + "=" * 60)
print("SECTION 6: STATISTICAL TESTING")
print("=" * 60)

# Test 1 - ANOVA
print("\n--- Test 1: ANOVA (Amount across Occupation groups) ---")
print("H0: Mean Amount is equal across all occupation groups")
print("HA: At least one occupation group has a different mean Amount")
groups = [group['Amount'].dropna().values for _, group in df.groupby('Occupation')]
stat, p_value = f_oneway(*groups)
print(f"\nF-statistic : {stat:.4f}")
print(f"P-value     : {p_value:.6f}")
if p_value < 0.05:
    print("Conclusion  : Reject H0 -- significant difference in Amount across occupation groups (p < 0.05)")
else:
    print("Conclusion  : Fail to reject H0 -- no significant difference found (p >= 0.05)")

# Group means
print("\nMean Amount per Occupation:")
print(df.groupby('Occupation')['Amount'].mean().round(2).sort_values(ascending=False))

# Test 2 - Chi-Square
print("\n--- Test 2: Chi-Square (Product_Category vs Zone) ---")
print("H0: Product_Category and Zone are independent")
print("HA: There is an association between Product_Category and Zone")
contingency = pd.crosstab(df['Product_Category'], df['Zone'])
print("\nContingency table:")
print(contingency)
chi2, p_value2, dof, expected = chi2_contingency(contingency)
print(f"\nChi2 statistic : {chi2:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"P-value        : {p_value2:.6f}")
if p_value2 < 0.05:
    print("Conclusion     : Reject H0 -- Product_Category and Zone are associated (p < 0.05)")
else:
    print("Conclusion     : Fail to reject H0 -- no significant association found (p >= 0.05)")

print("\n" + "=" * 60)
print("ALL SECTIONS COMPLETE - check outputs/ folder for plots")
print("=" * 60)
