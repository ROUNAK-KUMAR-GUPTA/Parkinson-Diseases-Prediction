import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              roc_curve, auc, precision_recall_curve, f1_score,
                              precision_score, recall_score)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from xgboost import XGBClassifier
import joblib
import json
import os

# ─── Color Palette ──────────────────────────────────────────────────────────────
COLORS = {
    'primary':   '#4C72B0',
    'secondary': '#DD8452',
    'healthy':   '#2ecc71',
    'parkinsons':'#e74c3c',
    'accent':    '#9b59b6',
    'bg':        '#f8f9fa',
    'dark':      '#2c3e50',
}
PALETTE = [COLORS['primary'], COLORS['secondary'], '#55A868', '#C44E52',
           '#8172B2', '#937860', '#DA8BC3', '#8C8C8C']

sns.set_theme(style='whitegrid', palette=PALETTE)
plt.rcParams.update({'figure.facecolor': COLORS['bg'], 'axes.facecolor': COLORS['bg'],
                     'font.family': 'DejaVu Sans', 'axes.titlesize': 13,
                     'axes.labelsize': 11})

os.makedirs('visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & EXPLORE DATA
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  PARKINSON'S DISEASE PREDICTION – VOICE ANALYSIS PIPELINE")
print("=" * 65)

df = pd.read_csv('data/parkinsons.csv')
df.columns = df.columns.str.strip()

print(f"\n📊 Dataset shape : {df.shape}")
print(f"   Healthy (0)   : {(df['status']==0).sum()} samples")
print(f"   Parkinson's(1): {(df['status']==1).sum()} samples")
print(f"   Features       : {df.shape[1]-2}  (excluding name & status)")

# Feature groups
freq_features    = ['MDVP:Fo(Hz)','MDVP:Fhi(Hz)','MDVP:Flo(Hz)']
jitter_features  = ['MDVP:Jitter(%)','MDVP:Jitter(Abs)','MDVP:RAP','MDVP:PPQ','Jitter:DDP']
shimmer_features = ['MDVP:Shimmer','MDVP:Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','MDVP:APQ','Shimmer:DDA']
noise_features   = ['NHR','HNR']
nonlinear_features = ['RPDE','DFA','spread1','spread2','D2','PPE']
all_features     = freq_features + jitter_features + shimmer_features + noise_features + nonlinear_features

X = df[all_features]
y = df['status']

# ═══════════════════════════════════════════════════════════════════════════════
# 2. EDA VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 2a. Class Distribution ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Dataset Overview – Class Distribution", fontsize=15, fontweight='bold', color=COLORS['dark'])

counts = y.value_counts()
labels = ["Healthy", "Parkinson's"]
colors = [COLORS['healthy'], COLORS['parkinsons']]

axes[0].pie(counts, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, wedgeprops={'edgecolor':'white','linewidth':2},
            textprops={'fontsize':12})
axes[0].set_title("Class Balance", fontweight='bold')

axes[1].bar(labels, counts.values, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
for i, (v, c) in enumerate(zip(counts.values, colors)):
    axes[1].text(i, v + 1, str(v), ha='center', fontweight='bold', fontsize=13, color=c)
axes[1].set_ylabel("Count")
axes[1].set_title("Sample Counts", fontweight='bold')
axes[1].set_ylim(0, max(counts.values) * 1.2)

plt.tight_layout()
plt.savefig('visualizations/01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 01_class_distribution.png")

# ── 2b. Feature Correlation Heatmap ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 13))
corr = df[all_features + ['status']].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, ax=ax, annot_kws={'size': 7},
            linewidths=0.3, linecolor='white',
            cbar_kws={'shrink': 0.8})
ax.set_title("Feature Correlation Matrix", fontsize=16, fontweight='bold', pad=15, color=COLORS['dark'])
plt.tight_layout()
plt.savefig('visualizations/02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 02_correlation_heatmap.png")

# ── 2c. Feature Distributions by Class ──────────────────────────────────────
top_features = ['MDVP:Fo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Shimmer', 'HNR',
                'RPDE', 'DFA', 'spread1', 'PPE']

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Voice Feature Distributions: Healthy vs Parkinson's", fontsize=15, fontweight='bold', color=COLORS['dark'])
axes = axes.flatten()

for i, feat in enumerate(top_features):
    healthy_vals = df[df['status']==0][feat]
    park_vals    = df[df['status']==1][feat]
    axes[i].hist(healthy_vals, bins=20, alpha=0.7, color=COLORS['healthy'], label='Healthy', edgecolor='white')
    axes[i].hist(park_vals,    bins=20, alpha=0.7, color=COLORS['parkinsons'], label="Parkinson's", edgecolor='white')
    axes[i].set_title(feat, fontweight='bold', fontsize=10)
    axes[i].set_xlabel('Value', fontsize=9)
    axes[i].set_ylabel('Count', fontsize=9)
    axes[i].legend(fontsize=8)
    axes[i].tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('visualizations/03_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 03_feature_distributions.png")

# ── 2d. Box Plots by Feature Group ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Feature Group Analysis by Class", fontsize=15, fontweight='bold', color=COLORS['dark'])

groups = [
    (jitter_features[:4], "Jitter Features"),
    (shimmer_features[:4], "Shimmer Features"),
    (nonlinear_features[:4], "Nonlinear Features"),
]

for ax, (feats, title) in zip(axes, groups):
    data_h = df[df['status']==0][feats]
    data_p = df[df['status']==1][feats]
    scaler_tmp = MinMaxScaler()
    data_h_norm = pd.DataFrame(scaler_tmp.fit_transform(data_h), columns=feats)
    data_p_norm = pd.DataFrame(scaler_tmp.transform(data_p), columns=feats)
    
    combined = pd.concat([
        data_h_norm.assign(Class='Healthy'),
        data_p_norm.assign(Class="Parkinson's")
    ]).melt(id_vars='Class', var_name='Feature', value_name='Value')
    
    sns.boxplot(data=combined, x='Feature', y='Value', hue='Class',
                palette={'Healthy': COLORS['healthy'], "Parkinson's": COLORS['parkinsons']},
                ax=ax, linewidth=1.2)
    ax.set_title(title, fontweight='bold')
    ax.set_xticklabels([f.split(':')[-1][:8] for f in feats], rotation=25, ha='right', fontsize=9)
    ax.set_xlabel('')

plt.tight_layout()
plt.savefig('visualizations/04_boxplots_by_group.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 04_boxplots_by_group.png")

# ── 2e. Violin Plots ─────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Violin Plots – Key Voice Features", fontsize=15, fontweight='bold', color=COLORS['dark'])
axes = axes.flatten()

key_features = ['MDVP:Fo(Hz)', 'HNR', 'RPDE', 'DFA', 'PPE', 'spread1']
for i, feat in enumerate(key_features):
    plot_df = df[['status', feat]].copy()
    plot_df['Class'] = plot_df['status'].map({0: 'Healthy', 1: "Parkinson's"})
    sns.violinplot(data=plot_df, x='Class', y=feat,
                   palette={'Healthy': COLORS['healthy'], "Parkinson's": COLORS['parkinsons']},
                   ax=axes[i], inner='box', linewidth=1.2)
    axes[i].set_title(feat, fontweight='bold', fontsize=11)
    axes[i].set_xlabel('')
    axes[i].tick_params(labelsize=9)

plt.tight_layout()
plt.savefig('visualizations/05_violin_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 05_violin_plots.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("  PREPROCESSING")
print("─" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train set: {X_train_sc.shape[0]} samples")
print(f"Test  set: {X_test_sc.shape[0]} samples")

# Feature importance via SelectKBest
selector = SelectKBest(f_classif, k='all')
selector.fit(X_train_sc, y_train)
feat_scores = pd.Series(selector.scores_, index=all_features).sort_values(ascending=False)

# ── Feature Importance Plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7))
colors_bar = [COLORS['parkinsons'] if s > feat_scores.median() else COLORS['primary']
              for s in feat_scores.values]
bars = ax.barh(feat_scores.index[::-1], feat_scores.values[::-1],
               color=colors_bar[::-1], edgecolor='white', linewidth=0.8)
ax.set_xlabel("F-Score (ANOVA)", fontsize=12)
ax.set_title("Feature Importance – ANOVA F-Score", fontsize=14, fontweight='bold', color=COLORS['dark'])
ax.axvline(feat_scores.median(), color=COLORS['dark'], linestyle='--', alpha=0.6, label='Median')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('visualizations/06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 06_feature_importance.png")

# ── PCA Visualization ─────────────────────────────────────────────────────────
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_sc)

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1],
                     c=[COLORS['healthy'] if l == 0 else COLORS['parkinsons'] for l in y_train],
                     alpha=0.75, edgecolors='white', linewidth=0.5, s=80)
h_patch = mpatches.Patch(color=COLORS['healthy'], label='Healthy')
p_patch = mpatches.Patch(color=COLORS['parkinsons'], label="Parkinson's")
ax.legend(handles=[h_patch, p_patch], fontsize=11)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=12)
ax.set_title("PCA – 2D Projection of Voice Features", fontsize=14, fontweight='bold', color=COLORS['dark'])
plt.tight_layout()
plt.savefig('visualizations/07_pca_projection.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 07_pca_projection.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 65)
print("  MODEL TRAINING")
print("─" * 65)

models = {
    'SVM (RBF)':           SVC(kernel='rbf', C=10, gamma=0.01, probability=True, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                          eval_metric='logloss', random_state=42, verbosity=0),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=42),
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'KNN':                 KNeighborsClassifier(n_neighbors=5, weights='distance'),
    'Decision Tree':       DecisionTreeClassifier(max_depth=6, random_state=42),
}

results = {}
cv_results = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1] if hasattr(model, 'predict_proba') else None
    
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    cv   = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy')
    
    fpr, tpr, _ = roc_curve(y_test, y_proba) if y_proba is not None else (None, None, None)
    roc_auc = auc(fpr, tpr) if fpr is not None else None
    
    results[name] = {
        'accuracy': acc, 'precision': prec, 'recall': rec,
        'f1': f1, 'roc_auc': roc_auc, 'y_pred': y_pred,
        'y_proba': y_proba, 'fpr': fpr, 'tpr': tpr,
        'cv_mean': cv.mean(), 'cv_std': cv.std()
    }
    cv_results[name] = cv
    print(f"  {name:<22} Acc={acc:.3f}  F1={f1:.3f}  AUC={roc_auc:.3f}  CV={cv.mean():.3f}±{cv.std():.3f}")

# ── Best Model ────────────────────────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['f1'])
best_model = models[best_name]
print(f"\n🏆 Best Model: {best_name} (F1={results[best_name]['f1']:.3f})")

# Save best model and scaler
joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(all_features, 'models/feature_names.pkl')

# Save metrics as JSON
metrics_json = {k: {m: float(v) for m, v in vals.items()
                    if m not in ['y_pred','y_proba','fpr','tpr']}
                for k, vals in results.items()}
with open('reports/model_metrics.json', 'w') as f:
    json.dump(metrics_json, f, indent=2)

# ═══════════════════════════════════════════════════════════════════════════════
# 5. EVALUATION PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── 5a. Model Comparison Bar Chart ───────────────────────────────────────────
metrics_df = pd.DataFrame({
    name: [r['accuracy'], r['precision'], r['recall'], r['f1'], r['roc_auc']]
    for name, r in results.items()
}, index=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']).T

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(metrics_df))
w = 0.15
for i, (metric, color) in enumerate(zip(metrics_df.columns, PALETTE)):
    bars = ax.bar(x + i*w, metrics_df[metric], width=w, label=metric,
                  color=color, edgecolor='white', linewidth=0.8)

ax.set_xticks(x + 2*w)
ax.set_xticklabels(metrics_df.index, rotation=20, ha='right', fontsize=10)
ax.set_ylabel("Score", fontsize=12)
ax.set_ylim(0.5, 1.08)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight='bold', color=COLORS['dark'])
ax.legend(loc='lower right', fontsize=9, ncol=2)
ax.axhline(0.9, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
plt.tight_layout()
plt.savefig('visualizations/08_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Saved: 08_model_comparison.png")

# ── 5b. ROC Curves ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot([0,1],[0,1],'k--', alpha=0.4, label='Random (AUC=0.50)')
for (name, r), color in zip(results.items(), PALETTE):
    if r['fpr'] is not None:
        lw = 3 if name == best_name else 1.5
        ax.plot(r['fpr'], r['tpr'], color=color, linewidth=lw,
                label=f"{name} (AUC={r['roc_auc']:.3f})")

ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves – All Models", fontsize=14, fontweight='bold', color=COLORS['dark'])
ax.legend(loc='lower right', fontsize=9)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig('visualizations/09_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 09_roc_curves.png")

# ── 5c. Confusion Matrices ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Confusion Matrices – All Models", fontsize=15, fontweight='bold', color=COLORS['dark'])
axes = axes.flatten()

for i, (name, r) in enumerate(results.items()):
    cm = confusion_matrix(y_test, r['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', "Parkinson's"],
                yticklabels=['Healthy', "Parkinson's"],
                ax=axes[i], linewidths=1, linecolor='white',
                annot_kws={'size': 14, 'weight': 'bold'})
    axes[i].set_title(f"{name}\nAcc={r['accuracy']:.3f}", fontsize=10, fontweight='bold')
    axes[i].set_xlabel('Predicted', fontsize=9)
    axes[i].set_ylabel('Actual', fontsize=9)

axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig('visualizations/10_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 10_confusion_matrices.png")

# ── 5d. Cross-Validation Box Plots ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 6))
cv_data = [cv_results[name] for name in results.keys()]
bp = ax.boxplot(cv_data, patch_artist=True, notch=False,
                medianprops={'color': 'white', 'linewidth': 2.5})
for patch, color in zip(bp['boxes'], PALETTE):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)

ax.set_xticklabels(list(results.keys()), rotation=20, ha='right', fontsize=10)
ax.set_ylabel("5-Fold CV Accuracy", fontsize=12)
ax.set_title("Cross-Validation Performance Distribution", fontsize=14, fontweight='bold', color=COLORS['dark'])
ax.set_ylim(0.7, 1.05)
ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('visualizations/11_cross_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 11_cross_validation.png")

# ── 5e. Best Model Feature Importances ────────────────────────────────────────
if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=all_features).sort_values(ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors_imp = [COLORS['parkinsons'] if v > importances.median() else COLORS['primary']
                  for v in importances.values]
    importances.plot(kind='barh', ax=ax, color=colors_imp, edgecolor='white', linewidth=0.5)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Feature Importances – {best_name}", fontsize=14, fontweight='bold', color=COLORS['dark'])
    ax.axvline(importances.median(), color=COLORS['dark'], linestyle='--', alpha=0.5, label='Median')
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('visualizations/12_feature_importances_best.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Saved: 12_feature_importances_best.png")

# ── 5f. Precision-Recall Curves ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
for (name, r), color in zip(results.items(), PALETTE):
    if r['y_proba'] is not None:
        prec_c, rec_c, _ = precision_recall_curve(y_test, r['y_proba'])
        lw = 3 if name == best_name else 1.5
        ax.plot(rec_c, prec_c, color=color, linewidth=lw, label=name)

ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves", fontsize=14, fontweight='bold', color=COLORS['dark'])
ax.legend(loc='lower left', fontsize=9)
plt.tight_layout()
plt.savefig('visualizations/13_precision_recall_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 13_precision_recall_curves.png")

# ── 5g. Summary Radar Chart ───────────────────────────────────────────────────
top3 = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)[:3]
categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'polar': True})
fig.patch.set_facecolor(COLORS['bg'])
ax.set_facecolor(COLORS['bg'])

for (name, r), color in zip(top3, [COLORS['primary'], COLORS['secondary'], COLORS['accent']]):
    values = [r['accuracy'], r['precision'], r['recall'], r['f1'], r['roc_auc']]
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2.5, label=name)
    ax.fill(angles, values, color=color, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=11, fontweight='bold')
ax.set_ylim(0.6, 1.0)
ax.set_yticks([0.7, 0.8, 0.9, 1.0])
ax.set_yticklabels(['0.7', '0.8', '0.9', '1.0'], size=9)
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
ax.set_title("Top 3 Models – Performance Radar", fontsize=14, fontweight='bold',
             color=COLORS['dark'], pad=20)
plt.tight_layout()
plt.savefig('visualizations/14_radar_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 14_radar_chart.png")

# ── 5h. Scatter – PPE vs HNR ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Key Voice Biomarkers – Scatter Analysis", fontsize=14, fontweight='bold', color=COLORS['dark'])

for ax, (feat_x, feat_y) in zip(axes, [('PPE','HNR'),('spread1','RPDE')]):
    for label, lname, color, marker in [(0,'Healthy',COLORS['healthy'],'o'), (1,"Parkinson's",COLORS['parkinsons'],'s')]:
        mask = y == label
        ax.scatter(df.loc[mask, feat_x], df.loc[mask, feat_y],
                   c=color, label=lname,
                   alpha=0.7, edgecolors='white', linewidth=0.5, s=70,
                   marker=marker)
    ax.set_xlabel(feat_x, fontsize=11)
    ax.set_ylabel(feat_y, fontsize=11)
    ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/15_scatter_biomarkers.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: 15_scatter_biomarkers.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
summary = {
    'best_model': best_name,
    'best_accuracy': float(results[best_name]['accuracy']),
    'best_f1': float(results[best_name]['f1']),
    'best_auc': float(results[best_name]['roc_auc']),
    'best_cv': float(results[best_name]['cv_mean']),
    'top_features': feat_scores.head(5).index.tolist(),
    'all_models': {k: {m: float(v) for m, v in vals.items()
                       if m not in ['y_pred','y_proba','fpr','tpr']}
                   for k, vals in results.items()}
}
with open('reports/summary.json','w') as f:
    json.dump(summary, f, indent=2)

print(f"\n  Best Model  : {best_name}")
print(f"  Accuracy    : {results[best_name]['accuracy']:.4f}")
print(f"  F1 Score    : {results[best_name]['f1']:.4f}")
print(f"  AUC-ROC     : {results[best_name]['roc_auc']:.4f}")
print(f"  CV Accuracy : {results[best_name]['cv_mean']:.4f} ± {results[best_name]['cv_std']:.4f}")
print(f"\n  Top 5 Features: {feat_scores.head(5).index.tolist()}")
print(f"\n✅ Models saved to: models/")
print(f"✅ Plots  saved to: visualizations/")
print(f"✅ Report saved to: reports/")
print("\n" + "=" * 65)