import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp, chi2_contingency, entropy, pointbiserialr
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.metrics import mean_squared_error


def test_onehot_fields(real_df, synth_df, fields):
    """
    Тестирование полей, которые должны быть one-hot закодированы
    """
    results = {}
    
    for field in fields:
        if field in real_df.columns and field in synth_df.columns:
            print('test')
            print(f"\n{'='*60}")
            print(f"ANALYSIS FOR: {field}")
            print(f"{'='*60}")
            
            # Базовые статистики
            real_unique = real_df[field].nunique()
            synth_unique = synth_df[field].nunique()
            
            real_counts = real_df[field].value_counts(normalize=True).head(10)
            synth_counts = synth_df[field].value_counts(normalize=True).head(10)
            
            # Расчет покрытия категорий
            real_categories = set(real_df[field].unique())
            synth_categories = set(synth_df[field].unique())
            coverage = len(real_categories & synth_categories) / max(1, len(real_categories)) * 100
            
            # Тест хи-квадрат для распределений
            all_categories = sorted(list(real_categories | synth_categories))
            contingency = pd.DataFrame({
                'real': [real_df[field].value_counts().get(cat, 0) for cat in all_categories],
                'synth': [synth_df[field].value_counts().get(cat, 0) for cat in all_categories]
            })
            
            try:
                chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency.values)
            except:
                chi2_stat, chi2_p, dof, expected = None, None, None, None
            
            # Энтропия распределений
            real_entropy = entropy(real_df[field].value_counts(normalize=True))
            synth_entropy = entropy(synth_df[field].value_counts(normalize=True))
            
            results[field] = {
                'real_unique_count': real_unique,
                'synth_unique_count': synth_unique,
                'category_coverage_percent': coverage,
                'chi2_statistic': chi2_stat,
                'chi2_pvalue': chi2_p,
                'real_entropy': real_entropy,
                'synth_entropy': synth_entropy,
                'entropy_difference': abs(real_entropy - synth_entropy),
                'distribution_similar': chi2_p > 0.05 if chi2_p is not None else None
            }
            
            # Визуализация
            try:
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                
                # Top categories comparison
                top_n = min(10, len(real_counts), len(synth_counts))
                if top_n > 0:
                    x = np.arange(top_n)
                    width = 0.35
                    
                    axes[0].bar(x - width/2, real_counts.values[:top_n], width, 
                               label='Real', alpha=0.7, color='blue')
                    axes[0].bar(x + width/2, [synth_counts.get(cat, 0) for cat in real_counts.index[:top_n]], 
                               width, label='Synthetic', alpha=0.7, color='orange')
                    axes[0].set_xticks(x)
                    axes[0].set_xticklabels(real_counts.index[:top_n], rotation=45, ha='right')
                    axes[0].set_title(f'Top {top_n} Categories: {field}')
                    axes[0].legend()
                    axes[0].set_ylabel('Proportion')
                else:
                    axes[0].text(0.5, 0.5, 'No data', ha='center', va='center')
                    axes[0].set_title(f'Top Categories: {field}')
                
                # Category coverage
                coverage_data = [len(real_categories - synth_categories), 
                               len(real_categories & synth_categories),
                               len(synth_categories - real_categories)]
                if sum(coverage_data) > 0:
                    labels = ['Only in Real', 'In Both', 'Only in Synthetic']
                    colors = ['red', 'green', 'orange']
                    axes[1].pie(coverage_data, labels=labels, colors=colors, autopct='%1.1f%%')
                    axes[1].set_title(f'Category Coverage: {field}')
                else:
                    axes[1].text(0.5, 0.5, 'No data', ha='center', va='center')
                    axes[1].set_title(f'Category Coverage: {field}')
                
                # Entropy comparison
                axes[2].bar(['Real', 'Synthetic'], [real_entropy, synth_entropy], 
                           color=['blue', 'orange'], alpha=0.7)
                axes[2].set_title(f'Entropy Comparison: {field}')
                axes[2].set_ylabel('Entropy')
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Visualization error for {field}: {e}")
            
            # Вывод результатов
            print(f"Unique values: Real={real_unique}, Synthetic={synth_unique}")
            print(f"Category coverage: {coverage:.1f}%")
            if chi2_p is not None:
                print(f"Chi-square test: p-value = {chi2_p:.4f} ({'Similar' if chi2_p > 0.05 else 'Different'})")
            print(f"Entropy: Real={real_entropy:.3f}, Synthetic={synth_entropy:.3f}")
            if len(real_counts) > 0:
                print(f"Top 3 real categories: {list(real_counts.index[:min(3, len(real_counts))])}")
            if len(synth_counts) > 0:
                print(f"Top 3 synth categories: {list(synth_counts.index[:min(3, len(synth_counts))])}")
    
    return results


if __name__ == "__main__":
    ONEHOT_FIELDS = ['PaymentSystem', 'OpType', 'DETAILEDCARDTYPE', 'TRANMETHOD', 'ISOWNTERMINAL', 
                 'NAME', 'MCC', 'DEVICETYPE', 'CurrencyName', 'CITY', 'REGION', 'ISVIRTUALTRANSACTION']
    real_data = pd.read_csv('Data/Sber/Clear/transact_1.csv')  # Реальные данные
    synth_data = pd.read_csv('Data/Sber/Clear/transact_400000_samples.csv')
    onehot_results = test_onehot_fields(real_data, synth_data, ONEHOT_FIELDS)
    print("test")