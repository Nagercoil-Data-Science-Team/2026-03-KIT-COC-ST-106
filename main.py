import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 18
plt.rcParams['font.weight'] = 'bold'
dataset_path = r"Detailed-Financials-Data-Of-4456-NSE-_-BSE-Company"   # change to your dataset folder

data_list = []

for company in os.listdir(dataset_path):

    company_folder = os.path.join(dataset_path, company)

    if os.path.isdir(company_folder):

        try:

            # ---------- SHAREHOLDING ----------
            shareholding = pd.read_csv(
                os.path.join(company_folder, "Yearly_Shareholding_Pattern.csv"),
                index_col=0
            )

            # transpose because rows = variables
            shareholding = shareholding.T

            latest_share = shareholding.iloc[-1]

            promoter = latest_share.get("Promoters", np.nan)
            fii = latest_share.get("FIIs", np.nan)
            dii = latest_share.get("DIIs", np.nan)
            public = latest_share.get("Public", np.nan)
            shareholders = latest_share.get("No. of Shareholders", np.nan)

            # ---------- RATIOS ----------
            ratios = pd.read_csv(os.path.join(company_folder, "Ratios.csv"), index_col=0)
            ratios = ratios.T
            latest_ratio = ratios.iloc[-1]

            debtor_days = latest_ratio.get("Debtor Days", np.nan)
            inventory_days = latest_ratio.get("Inventory Days", np.nan)
            days_payable = latest_ratio.get("Days Payable", np.nan)
            cash_conversion = latest_ratio.get("Cash Conversion Cycle", np.nan)
            working_capital_days = latest_ratio.get("Working Capital Days", np.nan)
            roce = latest_ratio.get("ROCE %", np.nan)

            # ---------- BALANCE SHEET ----------
            balance = pd.read_csv(
                os.path.join(company_folder, "Yearly_Balance_Sheet.csv"),
                index_col=0
            )

            balance = balance.T
            latest_balance = balance.iloc[-1]

            equity = latest_balance.get("Equity Capital", np.nan)
            reserves = latest_balance.get("Reserves", np.nan)
            borrowings = latest_balance.get("Borrowings", np.nan)
            other_liab = latest_balance.get("Other Liabilities", np.nan)
            total_liab = latest_balance.get("Total Liabilities", np.nan)

            fixed_assets = latest_balance.get("Fixed Assets", np.nan)
            cwip = latest_balance.get("CWIP", np.nan)
            investments = latest_balance.get("Investments", np.nan)
            other_assets = latest_balance.get("Other Assets", np.nan)
            total_assets = latest_balance.get("Total Assets", np.nan)

            # ---------- CASH FLOW ----------
            cashflow = pd.read_csv(
                os.path.join(company_folder, "Yearly_Cash_flow.csv"),
                index_col=0
            )

            cashflow = cashflow.T
            latest_cash = cashflow.iloc[-1]

            ocf = latest_cash.get("Cash from Operating Activity", np.nan)
            icf = latest_cash.get("Cash from Investing Activity", np.nan)
            fcf = latest_cash.get("Cash from Financing Activity", np.nan)
            net_cf = latest_cash.get("Net Cash Flow", np.nan)

            # ---------- STORE DATA ----------
            data_list.append([
                company,
                promoter, fii, dii, public, shareholders,
                debtor_days, inventory_days, days_payable,
                cash_conversion, working_capital_days,
                roce,
                equity, reserves, borrowings, other_liab,
                total_liab,
                fixed_assets, cwip, investments,
                other_assets, total_assets,
                ocf, icf, fcf, net_cf
            ])

        except Exception as e:
            print("Skipping:", company, "Reason:", e)


# ---------- CREATE DATAFRAME ----------
columns = [
    "Company",
    "Promoters","FIIs","DIIs","Public","Shareholders",
    "Debtor_Days","Inventory_Days","Days_Payable",
    "Cash_Conversion_Cycle","Working_Capital_Days",
    "ROCE",
    "Equity_Capital","Reserves","Borrowings",
    "Other_Liabilities","Total_Liabilities",
    "Fixed_Assets","CWIP","Investments",
    "Other_Assets","Total_Assets",
    "OCF","ICF","FCF","Net_Cash_Flow"
]

df = pd.DataFrame(data_list, columns=columns)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

# ---------------------------------------
# STEP 1 : DATA CLEANING
# ---------------------------------------

# Convert numeric columns
numeric_cols = df.columns.drop("Company")
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

# ---- Missing value handling (Median Imputation) ----
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# ---------------------------------------
# STEP 2 : OUTLIER REMOVAL (Z-SCORE)
# ---------------------------------------

z_scores = np.abs(stats.zscore(df[numeric_cols]))

df = df[(z_scores < 3).all(axis=1)]

print("Dataset size after outlier removal:", df.shape)

# ---------------------------------------
# STEP 3 : NORMALIZATION (MIN MAX)
# ---------------------------------------

scaler = MinMaxScaler()

scaled_features = scaler.fit_transform(df[numeric_cols])

df_scaled = pd.DataFrame(scaled_features, columns=numeric_cols)

df_scaled["Company"] = df["Company"].values

# ---------------------------------------
# STEP 4 : DECISION VARIABLES
# Convert Shareholding % → Proportion
# ---------------------------------------

df_scaled["x1"] = df_scaled["Promoters"] / 100
df_scaled["x2"] = df_scaled["FIIs"] / 100
df_scaled["x3"] = df_scaled["DIIs"] / 100
df_scaled["x4"] = df_scaled["Public"] / 100

# ---------------------------------------
# STEP 5 : ENFORCE CONSTRAINT
# x1 + x2 + x3 + x4 = 1
# ---------------------------------------

total = df_scaled["x1"] + df_scaled["x2"] + df_scaled["x3"] + df_scaled["x4"]

df_scaled["x1"] = df_scaled["x1"] / total
df_scaled["x2"] = df_scaled["x2"] / total
df_scaled["x3"] = df_scaled["x3"] / total
df_scaled["x4"] = df_scaled["x4"] / total

print(df_scaled[["x1","x2","x3","x4"]].head())

import numpy as np

# ------------------------------------------------
# STEP 3 : OWNERSHIP METRICS
# ------------------------------------------------

# HHI Calculation
df_scaled["HHI"] = (
    df_scaled["x1"]**2 +
    df_scaled["x2"]**2 +
    df_scaled["x3"]**2 +
    df_scaled["x4"]**2
)

# Entropy Calculation
df_scaled["Entropy"] = -(
    df_scaled["x1"] * np.log(df_scaled["x1"] + 1e-10) +
    df_scaled["x2"] * np.log(df_scaled["x2"] + 1e-10) +
    df_scaled["x3"] * np.log(df_scaled["x3"] + 1e-10) +
    df_scaled["x4"] * np.log(df_scaled["x4"] + 1e-10)
)

# Institutional Dominance Ratio
df_scaled["IDR"] = df_scaled["x2"] + df_scaled["x3"]

print(df_scaled[["Company","HHI","Entropy","IDR"]].head())

# ------------------------------------------------
# STEP 4 : CONSTRUCTION OF SCORES
# ------------------------------------------------

from sklearn.preprocessing import MinMaxScaler

# ---------- GOVERNANCE BALANCE SCORE ----------
alpha1 = 0.6
alpha2 = 0.4

df_scaled["Governance_Score"] = (
    alpha1 * df_scaled["Entropy"] +
    alpha2 * df_scaled["IDR"]
)


# ---------- PERFORMANCE SCORE ----------

performance_cols = ["ROCE"]  # add ROE, EPS, SG, PG if available

scaler = MinMaxScaler()
df_scaled[performance_cols] = scaler.fit_transform(df_scaled[performance_cols])

beta1 = 1.0  # since only ROCE currently available

df_scaled["Performance_Score"] = beta1 * df_scaled["ROCE"]


# ---------- STABILITY SCORE ----------

stability_cols = ["OCF","Net_Cash_Flow"]

df_scaled[stability_cols] = scaler.fit_transform(df_scaled[stability_cols])

gamma1 = 0.5
gamma2 = 0.5

df_scaled["Stability_Score"] = (
    gamma1 * df_scaled["OCF"] +
    gamma2 * df_scaled["Net_Cash_Flow"]
)


# ---------- REPUTATION SCORE ----------

w1 = 0.4
w2 = 0.3
w3 = 0.3

df_scaled["Reputation"] = (
    w1 * df_scaled["Performance_Score"] +
    w2 * df_scaled["Stability_Score"] +
    w3 * df_scaled["Governance_Score"]
)

print(df_scaled[[
    "Company",
    "Governance_Score",
    "Performance_Score",
    "Stability_Score",
    "Reputation"
]].head())


# ------------------------------------------------
# STEP 5 : ADJUSTED VOTING POWER MODEL
# ------------------------------------------------

# Governance credibility coefficients

df_scaled["theta_promoter"] = df_scaled["Reputation"]

df_scaled["theta_fii"] = df_scaled["Reputation"] + df_scaled["IDR"]

df_scaled["theta_dii"] = df_scaled["Reputation"] + df_scaled["IDR"]

df_scaled["theta_public"] = df_scaled["Reputation"]


# ---------- Adjusted Voting Power ----------

df_scaled["Promoter_AVP"] = df_scaled["x1"] * df_scaled["theta_promoter"]

df_scaled["FII_AVP"] = df_scaled["x2"] * df_scaled["theta_fii"]

df_scaled["DII_AVP"] = df_scaled["x3"] * df_scaled["theta_dii"]

df_scaled["Public_AVP"] = df_scaled["x4"] * df_scaled["theta_public"]


# ---------- Total AVP ----------

df_scaled["Total_AVP"] = (
    df_scaled["Promoter_AVP"] +
    df_scaled["FII_AVP"] +
    df_scaled["DII_AVP"] +
    df_scaled["Public_AVP"]
)


# ---------- Governance Influence ----------

df_scaled["Promoter_Influence"] = df_scaled["Promoter_AVP"] / df_scaled["Total_AVP"]

df_scaled["FII_Influence"] = df_scaled["FII_AVP"] / df_scaled["Total_AVP"]

df_scaled["DII_Influence"] = df_scaled["DII_AVP"] / df_scaled["Total_AVP"]

df_scaled["Public_Influence"] = df_scaled["Public_AVP"] / df_scaled["Total_AVP"]


print(df_scaled[[
    "Company",
    "Promoter_Influence",
    "FII_Influence",
    "DII_Influence",
    "Public_Influence"
]].head())


import hashlib
import json

# ------------------------------------------------
# STEP 6 : BLOCKCHAIN GOVERNANCE LAYER
# ------------------------------------------------


# ---------- HASH DATASET FOR INTEGRITY ----------

dataset_string = df_scaled.to_json()

dataset_hash = hashlib.sha256(dataset_string.encode()).hexdigest()

print("Dataset Hash:", dataset_hash)


# ---------- SMART CONTRACT RULES (SIMULATION) ----------

def equity_constraint(row):
    total = row["x1"] + row["x2"] + row["x3"] + row["x4"]
    return abs(total - 1) < 0.001


def governance_threshold(row):
    return row["Governance_Score"] > 0.3


def reputation_weighted_voting(row):
    return row["Reputation"] * (
        row["Promoter_Influence"] +
        row["FII_Influence"] +
        row["DII_Influence"] +
        row["Public_Influence"]
    )


# ---------- APPLY SMART CONTRACT VALIDATION ----------

df_scaled["Equity_Valid"] = df_scaled.apply(equity_constraint, axis=1)

df_scaled["Governance_Valid"] = df_scaled.apply(governance_threshold, axis=1)

df_scaled["Voting_Score"] = df_scaled.apply(reputation_weighted_voting, axis=1)


# ---------- CONSENSUS MECHANISM (Hybrid PBFT + DPoS Simulation) ----------

def hybrid_consensus(row):

    pbft_votes = [
        row["Promoter_Influence"],
        row["FII_Influence"],
        row["DII_Influence"],
        row["Public_Influence"]
    ]

    pbft_result = sum(pbft_votes) > 0.66   # PBFT supermajority

    dpos_validator = row["Reputation"] > 0.4   # Delegated validator

    return pbft_result and dpos_validator


df_scaled["Consensus_Approved"] = df_scaled.apply(hybrid_consensus, axis=1)


# ---------- BLOCK CREATION ----------

blockchain = []

for index, row in df_scaled.iterrows():

    block = {

        "Company": row["Company"],

        "Equity_Structure": {
            "Promoter": row["x1"],
            "FII": row["x2"],
            "DII": row["x3"],
            "Public": row["x4"]
        },

        "Governance_Score": row["Governance_Score"],
        "Performance_Score": row["Performance_Score"],
        "Stability_Score": row["Stability_Score"],

        "Reputation": row["Reputation"],

        "Adjusted_Voting_Power": {
            "Promoter": row["Promoter_AVP"],
            "FII": row["FII_AVP"],
            "DII": row["DII_AVP"],
            "Public": row["Public_AVP"]
        },

        "Consensus_Approved": row["Consensus_Approved"],

        "Dataset_Hash": dataset_hash

    }

    blockchain.append(block)


print("Blockchain blocks created:", len(blockchain))
import numpy as np
import matplotlib.pyplot as plt

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination


# ------------------------------------------------
# NSGA-II PROBLEM
# ------------------------------------------------

class EquityOptimization(Problem):

    def __init__(self):
        super().__init__(
            n_var=4,
            n_obj=4,
            n_constr=1,
            xl=0,
            xu=1
        )

    def _evaluate(self, X, out, *args, **kwargs):

        x1 = X[:,0]
        x2 = X[:,1]
        x3 = X[:,2]
        x4 = X[:,3]

        total = x1 + x2 + x3 + x4

        # ----------------------------
        # Ownership concentration
        # ----------------------------

        hhi = x1**2 + x2**2 + x3**2 + x4**2

        # ----------------------------
        # Ownership entropy
        # ----------------------------

        entropy = -(x1*np.log(x1+1e-10) +
                    x2*np.log(x2+1e-10) +
                    x3*np.log(x3+1e-10) +
                    x4*np.log(x4+1e-10))

        # ----------------------------
        # Institutional dominance
        # ----------------------------

        idr = x2 + x3

        governance = 0.6*entropy + 0.4*idr

        # ----------------------------
        # PERFORMANCE & STABILITY
        # ----------------------------

        pop_size = X.shape[0]

        performance_mean = np.mean(df_scaled["Performance_Score"])
        stability_mean = np.mean(df_scaled["Stability_Score"])

        performance = np.full(pop_size, performance_mean)
        stability = np.full(pop_size, stability_mean)

        # ----------------------------
        # OBJECTIVES
        # ----------------------------

        f1 = -governance
        f2 = -performance
        f3 = -stability
        f4 = hhi

        out["F"] = np.column_stack([f1,f2,f3,f4])

        # constraint: shares must sum to 1
        out["G"] = np.column_stack([total - 1])


# ------------------------------------------------
# RUN OPTIMIZATION
# ------------------------------------------------

problem = EquityOptimization()

algorithm = NSGA2(pop_size=100)

termination = get_termination("n_gen",100)

result = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    verbose=True
)

solutions = result.X
objectives = result.F

print("Solutions:",solutions.shape)


# ------------------------------------------------
# PLOT 1 : PARETO FRONT
# ------------------------------------------------

plt.figure(figsize=(8,6))
plt.scatter(-objectives[:,0], objectives[:,3],color="#612D53")
plt.xlabel("Governance Balance",fontweight="bold")
plt.ylabel("HHI",fontweight="bold")
plt.title("Pareto Front",fontweight="bold")
plt.savefig("pareto_front.png",dpi=800)
plt.show()


# ------------------------------------------------
# PLOT 2 : PERFORMANCE VS STABILITY
# ------------------------------------------------

plt.figure(figsize=(12,8))
plt.scatter(-objectives[:,1], -objectives[:,2],color="#41431B")
plt.xlabel("Performance Score",fontweight="bold")
plt.ylabel("Stability Score",fontweight="bold")
plt.title("Performance vs Stability",fontweight="bold")
plt.savefig("performance_vs_stability.png",dpi=800)
plt.show()


# ------------------------------------------------
# PLOT 3 : CONVERGENCE CURVE
# ------------------------------------------------

convergence = []

for gen in result.history:
    best = np.min(gen.pop.get("F")[:,3])
    convergence.append(best)

plt.figure()
plt.plot(convergence)
plt.xlabel("Generation")
plt.ylabel("Best HHI")
plt.title("NSGA-II Convergence")
plt.show()


# ------------------------------------------------
# PLOT 4 : OPTIMAL EQUITY STRUCTURE
# ------------------------------------------------

best_solution = solutions[0]

labels = ["Promoter","FII","DII","Public"]

plt.figure(figsize=(8,6))
plt.bar(labels,best_solution,color="#8E977D")
plt.xlabel("Shareholder Type",fontweight="bold")
plt.ylabel("Share Proportion",fontweight="bold")
plt.title("Optimized Equity Structure",fontweight="bold")
plt.savefig("equity_structure.png",dpi=800)
plt.show()
import seaborn as sns
import matplotlib.pyplot as plt

# =====================================================
# 1. DESCRIPTIVE STATISTICS OF DATASET
# =====================================================

plt.figure()
desc_stats = df_scaled.describe().T
plt.imshow(desc_stats, aspect='auto')
plt.colorbar()
plt.title("Descriptive Statistics of Dataset")
plt.yticks(range(len(desc_stats.index)), desc_stats.index)
plt.xticks(range(len(desc_stats.columns)), desc_stats.columns, rotation=45)
plt.tight_layout()
plt.show()


# =====================================================
# 2. OWNERSHIP METRIC SUMMARY
# =====================================================

plt.figure(figsize=(8,6))
ownership_metrics = df_scaled[["HHI","Entropy","IDR"]].mean()
plt.bar(ownership_metrics.index, ownership_metrics.values,color='#6B7445')
plt.title("Ownership Metric Summary",fontweight="bold")
plt.ylabel("Average Value",fontweight="bold")
plt.xlabel("Metric",fontweight="bold")
plt.show()


# =====================================================
# 3. SCORE DISTRIBUTION SUMMARY
# =====================================================

plt.figure(figsize=(12,8))
scores = df_scaled[[
    "Governance_Score",
    "Performance_Score",
    "Stability_Score",
    "Reputation"
]]

plt.boxplot(scores.values, labels=scores.columns)
plt.title("Score Distribution Summary",fontweight="bold")
plt.xlabel("Governance Score",fontweight="bold")
plt.ylabel("Score Value",fontweight="bold")
plt.savefig("scores.png",dpi=800)
plt.show()


# =====================================================
# 4. CORRELATION HEATMAP
# =====================================================

plt.figure(figsize=(15,8))
corr = df_scaled[[
    "Governance_Score",
    "Performance_Score",
    "Stability_Score",
    "Reputation",
    "HHI",
    "Entropy",
    "IDR"
]].corr()

sns.heatmap(corr, annot=True)
plt.title("Correlation Heatmap",fontweight="bold")
plt.xlabel("Governance Score",fontweight="bold")
plt.ylabel("Performance Score",fontweight="bold")
plt.savefig("correlation_heatmap.png",dpi=800)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=0)
plt.yticks(range(len(corr.index)), corr.index,rotation=0)
plt.show()


# =====================================================
# 5. OWNERSHIP DISTRIBUTION HISTOGRAM
# =====================================================

plt.figure(figsize=(12,8))
plt.hist(df_scaled["x1"], bins=30, alpha=0.5, label="Promoter")
plt.hist(df_scaled["x2"], bins=30, alpha=0.5, label="FII")
plt.hist(df_scaled["x3"], bins=30, alpha=0.5, label="DII")
plt.hist(df_scaled["x4"], bins=30, alpha=0.5, label="Public")

plt.legend()
plt.title("Ownership Distribution Histogram",fontweight="bold")
plt.xlabel("Ownership Proportion",fontweight="bold")
plt.ylabel("Frequency",fontweight="bold")
plt.savefig("ownership_distribution.png",dpi=800)
plt.show()


# =====================================================
# 6. ADJUSTED VOTING POWER DISTRIBUTION
# =====================================================

plt.figure(figsize=(9,6))
avp_means = [
    df_scaled["Promoter_AVP"].mean(),
    df_scaled["FII_AVP"].mean(),
    df_scaled["DII_AVP"].mean(),
    df_scaled["Public_AVP"].mean()
]

labels = ["Promoter","FII","DII","Public"]

plt.bar(labels, avp_means,color='#296374')
plt.title("Adjusted Voting Power Distribution",fontweight="bold")
plt.ylabel("Average Voting Power",fontweight="bold")
plt.xlabel("Shareholder Type",fontweight="bold")
plt.savefig("adjusted_voting_power.png",dpi=800)
plt.show()


# =====================================================
# 7. SMART CONTRACT VALIDATION ANALYSIS
# =====================================================

plt.figure(figsize=(14,8))

valid_counts = [
    df_scaled["Equity_Valid"].sum(),
    df_scaled["Governance_Valid"].sum(),
    df_scaled["Consensus_Approved"].sum()
]

labels = [
    "Equity Constraint Valid",
    "Governance Threshold Valid",
    "Consensus Approved"
]

plt.bar(labels, valid_counts,color='#1D546D')

plt.title("Smart Contract Validation Analysis",fontweight="bold")
plt.xlabel("Shareholder Type",fontweight="bold")
plt.ylabel("Number of Companies",fontweight="bold")
plt.xticks(rotation=0)
plt.savefig("smart_validation.png",dpi=800)
plt.show()

# =====================================================
# PERFORMANCE SCORE PLOT
# =====================================================

plt.figure(figsize=(8,6))

performance_values = df_scaled["Performance_Score"]

plt.hist(performance_values, bins=40, edgecolor='black')

plt.title("Performance Score Distribution", fontweight="bold")
plt.xlabel("Performance Score", fontweight="bold")
plt.ylabel("Number of Companies", fontweight="bold")
plt.savefig("performance_score.png",dpi=800)
plt.show()



# =====================================================
# STABILITY SCORE PLOT
# =====================================================

plt.figure(figsize=(8,6))

stability_values = df_scaled["Stability_Score"]

plt.hist(stability_values, bins=40, edgecolor='black')

plt.title("Stability Score Distribution", fontweight="bold")
plt.xlabel("Stability Score", fontweight="bold")
plt.ylabel("Number of Companies", fontweight="bold")
plt.savefig("stability_score.png",dpi=800)
plt.show()



# =====================================================
# GOVERNANCE BALANCE PLOT
# =====================================================

plt.figure(figsize=(8,6))

governance_values = df_scaled["Governance_Score"]

plt.hist(governance_values, bins=40, edgecolor='black')

plt.title("Governance Balance Distribution", fontweight="bold")
plt.xlabel("Governance Score", fontweight="bold")
plt.ylabel("Number of Companies", fontweight="bold")
plt.savefig("governance_score.png",dpi=800)
plt.show()



# =====================================================
# CORRECT CONVERGENCE METRIC
# =====================================================

convergence = []

for gen in result.history:
    F = gen.pop.get("F")
    best_hhi = np.min(F[:,3])
    convergence.append(best_hhi)

plt.figure(figsize=(8,6))
plt.plot(convergence, marker='o')
plt.title("Optimization Convergence Metric", fontweight="bold")
plt.xlabel("Generation", fontweight="bold")
plt.ylabel("Best HHI Value", fontweight="bold")
plt.grid(True)
plt.show()


# =====================================================
# CORRECT HYPERVOLUME INDICATOR
# =====================================================

from pymoo.indicators.hv import HV

ref_point = np.array([1,1,1,1])
hv = HV(ref_point=ref_point)

hv_values = []

for gen in result.history:
    F = gen.pop.get("F")
    hv_values.append(hv(F))

plt.figure(figsize=(8,6))
plt.plot(hv_values, marker='o')
plt.title("Hypervolume Indicator (HV)", fontweight="bold")
plt.xlabel("Generation", fontweight="bold")
plt.ylabel("Hypervolume", fontweight="bold")
plt.grid(True)
plt.show()

# =====================================================
# PERFORMANCE SCORE DISTRIBUTION (SEPARATE WINDOW)
# =====================================================

plt.figure(figsize=(8,6))

plt.hist(df_scaled["Performance_Score"], bins=40, edgecolor="black")

plt.title("Performance Score Distribution", fontweight="bold")
plt.xlabel("Performance Score", fontweight="bold")
plt.ylabel("Number of Companies", fontweight="bold")

plt.grid(True)

plt.show()



# =====================================================
# STABILITY SCORE DISTRIBUTION (SEPARATE WINDOW)
# =====================================================

plt.figure(figsize=(8,6))

plt.hist(df_scaled["Stability_Score"], bins=40, edgecolor="black")

plt.title("Stability Score Distribution", fontweight="bold")
plt.xlabel("Stability Score", fontweight="bold")
plt.ylabel("Number of Companies", fontweight="bold")

plt.grid(True)

plt.show()



# =====================================================
# COMBINED SCORE DISTRIBUTION PLOTS
# Performance + Stability + Governance
# =====================================================

# =====================================================
# COMBINED SCORE DISTRIBUTION (BAR PLOT FORMAT)
# Performance + Stability + Governance
# =====================================================

plt.figure(figsize=(8,6))

scores = [
    df_scaled["Performance_Score"].mean(),
    df_scaled["Stability_Score"].mean(),
    df_scaled["Governance_Score"].mean()
]

labels = [
    "Performance Score",
    "Stability Score",
    "Governance Score"
]

colors = ["#296374", "#94A378", "#57595B"]

plt.bar(labels, scores, color=colors, edgecolor="black")

plt.title("Average Score Comparison", fontweight="bold")
plt.xlabel("Score Type", fontweight="bold")
plt.ylabel("Average Score Value", fontweight="bold")

plt.tight_layout()

plt.savefig("score_bar_plot.png", dpi=800)

plt.show()

# =====================================================
# DESCRIPTIVE STATISTICS FOR FIRST 5 COMPANIES
# =====================================================

print("\n==============================")
print("DESCRIPTIVE STATISTICS (FIRST 5 COMPANIES)")
print("==============================\n")

sample_companies = df_scaled.head(5)

desc_stats = sample_companies.describe().T

print(desc_stats)

print("\n==============================")
print("DATA FOR FIRST 5 COMPANIES")
print("==============================\n")

print(df_scaled.head(5))

# =====================================================
# SAVE COMPLETE DATASET TO EXCEL
# =====================================================

output_file = "Corporate_Governance_Analysis.xlsx"

df_scaled.to_excel(output_file, index=False)

print("\nDataset successfully saved to Excel file:", output_file)