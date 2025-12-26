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


def shorten_category(cat, max_len=15, ellipsis='...'):
    if isinstance(cat, str) and len(cat) > max_len:
        return cat[:max_len - len(ellipsis)] + ellipsis
    return str(cat)

def test_onehot_fields(real_df, synth_df, fields, max_label_length=15, truncate_ellipsis='...'):

    results = {}
    
    for field in fields:
        if field not in real_df.columns or field not in synth_df.columns:
            print(f"Field '{field}' not found in both datasets. Skipping.")
            continue
            
        print(f"\n{'='*60}")
        print(f"ANALYSIS FOR: {field}")
        print(f"{'='*60}")
        
        real_unique = real_df[field].nunique()
        synth_unique = synth_df[field].nunique()
        
        # Получаем распределения (нормализованные)
        real_counts = real_df[field].value_counts(normalize=True).head(10)
        synth_counts = synth_df[field].value_counts(normalize=True).head(10)
        
        real_categories = set(real_df[field].unique())
        synth_categories = set(synth_df[field].unique())
        coverage = len(real_categories & synth_categories) / max(1, len(real_categories)) * 100
        
        all_categories = sorted(list(real_categories | synth_categories))
        contingency = pd.DataFrame({
            'real': [real_df[field].value_counts().get(cat, 0) for cat in all_categories],
            'synth': [synth_df[field].value_counts().get(cat, 0) for cat in all_categories]
        })
        
        try:
            chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency.values)
        except Exception as e:
            chi2_stat, chi2_p, dof, expected = None, None, None, None
            print(f"Chi-square test failed for {field}: {e}")
        
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
        
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            top_n = min(10, len(real_counts), len(synth_counts))
            
            if top_n > 0:
                x = np.arange(top_n)
                width = 0.35
                
                # Сокращаем метки для графика
                real_labels = [shorten_category(cat, max_label_length, truncate_ellipsis) for cat in real_counts.index[:top_n]]
                synth_values = [synth_counts.get(cat, 0) for cat in real_counts.index[:top_n]]
                
                axes[0].bar(x - width/2, real_counts.values[:top_n], width, 
                           label='Real', alpha=0.7, color='blue')
                axes[0].bar(x + width/2, synth_values, width, 
                           label='Synthetic', alpha=0.7, color='orange')
                axes[0].set_xticks(x)
                axes[0].set_xticklabels(real_labels, rotation=45, ha='right', fontsize=9)
                axes[0].set_title(f'Top {top_n} Categories: {field}')
                axes[0].legend()
                axes[0].set_ylabel('Proportion')
                axes[0].grid(True, alpha=0.2)
            else:
                axes[0].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[0].set_title(f'Top Categories: {field}')
            
            # Круговая диаграмма покрытия
            coverage_data = [
                len(real_categories - synth_categories),  # Только в реальных
                len(real_categories & synth_categories),  # В обоих
                len(synth_categories - real_categories)   # Только в синтетических
            ]
            if sum(coverage_data) > 0:
                labels = ['Only in Real', 'In Both', 'Only in Synthetic']
                colors = ['red', 'green', 'orange']
                wedges, texts, autotexts = axes[1].pie(
                    coverage_data, labels=labels, colors=colors, autopct='%1.1f%%',
                    startangle=90
                )
                axes[1].set_title(f'Category Coverage: {field}')
                # Уменьшаем размер текста внутри секторов
                for autotext in autotexts:
                    autotext.set_fontsize(9)
            else:
                axes[1].text(0.5, 0.5, 'No data', ha='center', va='center')
                axes[1].set_title(f'Category Coverage: {field}')
            
            # Сравнение энтропии
            axes[2].bar(['Real', 'Synthetic'], [real_entropy, synth_entropy], 
                       color=['blue', 'orange'], alpha=0.7, edgecolor='black')
            axes[2].set_title(f'Entropy Comparison: {field}')
            axes[2].set_ylabel('Entropy')
            axes[2].grid(True, alpha=0.3)
            axes[2].set_ylim(0, max(real_entropy, synth_entropy) * 1.1)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Visualization error for {field}: {e}")
        
        # --- Вывод результатов ---
        print(f"Unique values: Real={real_unique}, Synthetic={synth_unique}")
        print(f"Category coverage: {coverage:.1f}%")
        if chi2_p is not None:
            status = 'Similar' if chi2_p > 0.05 else 'Different'
            print(f"Chi-square test: p-value = {chi2_p:.4f} ({status})")
        else:
            print("Chi-square test: Not applicable (insufficient data)")
        print(f"Entropy: Real={real_entropy:.3f}, Synthetic={synth_entropy:.3f}")
        
        if len(real_counts) > 0:
            top_real_short = [shorten_category(cat, 20, '...') for cat in real_counts.index[:min(3, len(real_counts))]]
            print(f"Top 3 real categories: {top_real_short}")
        if len(synth_counts) > 0:
            top_synth_short = [shorten_category(cat, 20, '...') for cat in synth_counts.index[:min(3, len(synth_counts))]]
            print(f"Top 3 synth categories: {top_synth_short}")
    
def test_categorical_fields(real_df, synth_df, fields):
    """
    Тестирование полей с высокой кардинальностью (ACCOUNT_ID, TERMINAL_CODE и т.д.)
    """
    results = {}
    
    for field in fields:
        if field in real_df.columns and field in synth_df.columns:
            print(f"\n{'='*60}")
            print(f"HIGH CARDINALITY ANALYSIS: {field}")
            print(f"{'='*60}")
            
            # Статистики уникальности
            real_unique = real_df[field].nunique()
            synth_unique = synth_df[field].nunique()
            
            # Частота появления значений
            real_freq = real_df[field].value_counts()
            synth_freq = synth_df[field].value_counts()
            
            # Распределение частот появления
            real_freq_dist = real_freq.value_counts(normalize=True)
            synth_freq_dist = synth_freq.value_counts(normalize=True)
            
            # Проверка на повторяемость паттернов
            overlap = len(set(real_df[field]) & set(synth_df[field]))
            
            # Анализ новых значений
            new_in_synth = len(set(synth_df[field]) - set(real_df[field]))
            
            results[field] = {
                'real_unique_count': real_unique,
                'synth_unique_count': synth_unique,
                'overlap_count': overlap,
                'new_in_synth_count': new_in_synth,
                'overlap_percentage': overlap / max(1, real_unique) * 100,
                'new_percentage': new_in_synth / max(1, synth_unique) * 100,
                'real_mean_frequency': real_freq.mean() if len(real_freq) > 0 else 0,
                'synth_mean_frequency': synth_freq.mean() if len(synth_freq) > 0 else 0,
                'real_freq_entropy': entropy(real_freq_dist) if len(real_freq_dist) > 0 else 0,
                'synth_freq_entropy': entropy(synth_freq_dist) if len(synth_freq_dist) > 0 else 0
            }
            
            try:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                
                if len(real_freq) > 0 and len(synth_freq) > 0:
                    max_freq = min(50, max(real_freq.max(), synth_freq.max()))
                    bins = np.arange(0, max_freq + 1, max(1, max_freq//20))
                    
                    axes[0, 0].hist(real_freq.values, bins=bins, alpha=0.7, 
                                   label=f'Real (n={real_unique})', density=True)
                    axes[0, 0].hist(synth_freq.values, bins=bins, alpha=0.7, 
                                   label=f'Synth (n={synth_unique})', density=True)
                    axes[0, 0].set_xlabel('Frequency of occurrence')
                    axes[0, 0].set_ylabel('Density')
                    axes[0, 0].set_title(f'Frequency Distribution: {field}')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                else:
                    axes[0, 0].text(0.5, 0.5, 'No data', ha='center', va='center')
                    axes[0, 0].set_title(f'Frequency Distribution: {field}')
                
                top_n = min(20, len(real_freq), len(synth_freq))
                if top_n > 0:
                    real_top = real_freq.head(top_n)
                    synth_top = synth_freq.head(top_n)
                    
                    x = np.arange(top_n)
                    width = 0.35
                    
                    axes[0, 1].bar(x - width/2, real_top.values, width, label='Real', alpha=0.7)
                    axes[0, 1].bar(x + width/2, synth_top.values, width, label='Synthetic', alpha=0.7)
                    axes[0, 1].set_xlabel('Rank')
                    axes[0, 1].set_ylabel('Frequency')
                    axes[0, 1].set_title(f'Top {top_n} Most Frequent Values: {field}')
                    axes[0, 1].legend()
                    axes[0, 1].set_xticks(x)
                    axes[0, 1].set_xticklabels(range(1, top_n+1))
                else:
                    axes[0, 1].text(0.5, 0.5, 'No data', ha='center', va='center')
                    axes[0, 1].set_title(f'Top Frequencies: {field}')
                
                categories = ['Real Only', 'Both', 'Synthetic Only']
                counts = [real_unique - overlap, overlap, new_in_synth]
                colors = ['blue', 'green', 'orange']
                
                axes[1, 0].bar(categories, counts, color=colors, alpha=0.7)
                axes[1, 0].set_title(f'Unique Values Analysis: {field}')
                axes[1, 0].set_ylabel('Count')
                for i, (cat, count) in enumerate(zip(categories, counts)):
                    axes[1, 0].text(i, count + max(counts)*0.02, str(count), 
                                   ha='center', va='bottom')
                
                if len(real_freq) > 1 and len(synth_freq) > 1:
                    axes[1, 1].loglog(np.arange(1, len(real_freq)+1), real_freq.values, 
                                    'o-', alpha=0.7, label='Real', markersize=4)
                    axes[1, 1].loglog(np.arange(1, len(synth_freq)+1), synth_freq.values, 
                                    's-', alpha=0.7, label='Synthetic', markersize=4)
                    axes[1, 1].set_xlabel('Rank (log)')
                    axes[1, 1].set_ylabel('Frequency (log)')
                    axes[1, 1].set_title(f'Zipf Plot: {field}')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                else:
                    axes[1, 1].text(0.5, 0.5, 'Not enough data', ha='center', va='center')
                    axes[1, 1].set_title(f'Zipf Plot: {field}')
                
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Visualization error for {field}: {e}")
            
            print(f"Unique values: Real={real_unique:,}, Synthetic={synth_unique:,}")
            print(f"Overlap: {overlap:,} values ({overlap/max(1, real_unique)*100:.1f}% of real)")
            print(f"New in synthetic: {new_in_synth:,} values ({new_in_synth/max(1, synth_unique)*100:.1f}% of synth)")
            print(f"Mean frequency: Real={results[field]['real_mean_frequency']:.2f}, "
                  f"Synthetic={results[field]['synth_mean_frequency']:.2f}")
            


def test_numerical_fields(real_df, synth_df, fields):
    """
    Детальное тестирование числовых полей
    """
    results = {}
    
    for field in fields:
        if field in real_df.columns and field in synth_df.columns:
            print(f"\n{'='*60}")
            print(f"NUMERICAL ANALYSIS: {field}")
            print(f"{'='*60}")
            
            real_data = real_df[field].dropna()
            synth_data = synth_df[field].dropna()
            
            if len(real_data) == 0 or len(synth_data) == 0:
                print(f"No valid data for {field}")
                continue
            
            # Базовые статистики
            stats_dict = {
                'count': [len(real_data), len(synth_data)],
                'mean': [real_data.mean(), synth_data.mean()],
                'std': [real_data.std(), synth_data.std()],
                'min': [real_data.min(), synth_data.min()],
                '25%': [real_data.quantile(0.25), synth_data.quantile(0.25)],
                '50%': [real_data.median(), synth_data.median()],
                '75%': [real_data.quantile(0.75), synth_data.quantile(0.75)],
                'max': [real_data.max(), synth_data.max()],
                'skewness': [real_data.skew(), synth_data.skew()],
                'kurtosis': [real_data.kurtosis(), synth_data.kurtosis()]
            }
            
            stats_df = pd.DataFrame(stats_dict, index=['Real', 'Synthetic'])
            
            # Статистические тесты
            try:
                ks_stat, ks_p = ks_2samp(real_data, synth_data)
                wasserstein_dist = wasserstein_distance(real_data, synth_data)
            except Exception as e:
                ks_stat, ks_p, wasserstein_dist = None, None, None
                print(f"Statistical tests failed: {e}")
            
            # Процентные различия
            real_mean = stats_df.loc['Real', 'mean']
            synth_mean = stats_df.loc['Synthetic', 'mean']
            
            percent_diff = {
                'mean_diff_pct': abs(real_mean - synth_mean) / max(abs(real_mean), 1e-10) * 100 if real_mean != 0 else None,
                'std_diff_pct': abs(stats_df.loc['Real', 'std'] - stats_df.loc['Synthetic', 'std']) / max(abs(stats_df.loc['Real', 'std']), 1e-10) * 100,
                'median_diff_pct': abs(stats_df.loc['Real', '50%'] - stats_df.loc['Synthetic', '50%']) / max(abs(stats_df.loc['Real', '50%']), 1e-10) * 100
            }
            
            results[field] = {
                'basic_statistics': stats_df.to_dict(),
                'ks_test': {'statistic': ks_stat, 'p_value': ks_p},
                'wasserstein_distance': wasserstein_dist,
                'percent_differences': percent_diff,
                'distributions_similar': ks_p > 0.05 if ks_p is not None else None
            }
            
            # Визуализация
            try:
                fig, axes = plt.subplots(2, 2, figsize=(18, 12))
                
                # Гистограммы
                bins = min(50, len(real_data) // 10, len(synth_data) // 10)
                bins = max(bins, 10)
                
                axes[0, 0].hist(real_data, bins=bins, alpha=0.7, density=True, 
                               label='Real', color='blue', edgecolor='black')
                axes[0, 0].hist(synth_data, bins=bins, alpha=0.7, density=True, 
                               label='Synthetic', color='orange', edgecolor='black')
                axes[0, 0].set_xlabel(field)
                axes[0, 0].set_ylabel('Density')
                axes[0, 0].set_title(f'Histogram: {field}')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Box plot
                bp = axes[0, 1].boxplot([real_data, synth_data], 
                                       labels=['Real', 'Synthetic'],
                                       patch_artist=True)
                bp['boxes'][0].set_facecolor('blue')
                bp['boxes'][1].set_facecolor('orange')
                axes[0, 1].set_ylabel(field)
                axes[0, 1].set_title(f'Box Plot: {field}')
                axes[0, 1].grid(True, alpha=0.3)
                

                
                # ECDF
                try:
                    from statsmodels.distributions.empirical_distribution import ECDF
                    ecdf_real = ECDF(real_data)
                    ecdf_synth = ECDF(synth_data)
                    
                    x = np.linspace(min(real_data.min(), synth_data.min()), 
                                  max(real_data.max(), synth_data.max()), 1000)
                    axes[1, 0].plot(x, ecdf_real(x), label='Real', linewidth=2)
                    axes[1, 0].plot(x, ecdf_synth(x), label='Synthetic', linewidth=2)
                    axes[1, 0].set_xlabel(field)
                    axes[1, 0].set_ylabel('Cumulative Probability')
                    axes[1, 0].set_title(f'ECDF: {field}')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                except:
                    axes[1, 0].text(0.5, 0.5, 'ECDF failed', ha='center', va='center')
                    axes[1, 0].set_title(f'ECDF: {field}')
                
                # Сравнение статистик
                metrics = ['mean', 'std', '25%', '50%', '75%']
                real_metrics = [stats_df.loc['Real', m] for m in metrics]
                synth_metrics = [stats_df.loc['Synthetic', m] for m in metrics]
                
                x_pos = np.arange(len(metrics))
                width = 0.35
                
                axes[1, 1].bar(x_pos - width/2, real_metrics, width, 
                              label='Real', alpha=0.7, color='blue')
                axes[1, 1].bar(x_pos + width/2, synth_metrics, width, 
                              label='Synthetic', alpha=0.7, color='orange')
                axes[1, 1].set_xticks(x_pos)
                axes[1, 1].set_xticklabels(metrics)
                axes[1, 1].set_ylabel('Value')
                axes[1, 1].set_title(f'Key Statistics: {field}')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3, axis='y')
                

                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"Visualization error for {field}: {e}")
            
            # Вывод результатов
            if ks_stat is not None:
                print(f"KS Test: statistic={ks_stat:.4f}, p-value={ks_p:.4f}")
            if wasserstein_dist is not None:
                print(f"Wasserstein Distance: {wasserstein_dist:.4f}")
            if percent_diff['mean_diff_pct'] is not None:
                print(f"Mean difference: {percent_diff['mean_diff_pct']:.1f}%")
            print(f"Std difference: {percent_diff['std_diff_pct']:.1f}%")
            print(f"Median difference: {percent_diff['median_diff_pct']:.1f}%")
            print(f"\nBasic Statistics:")
            print(stats_df.round(3))
    
def cramers_v(x, y):
    """Calculate Cramér's V statistic for categorial-categorial association"""
    confusion_matrix = pd.crosstab(x, y)
    try:
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
    except:
        return np.nan

def count_cramers(real_df, synth_df, fields):
    n_fields = len(fields)
    real_cramers_v = np.zeros((n_fields, n_fields))
    synth_cramers_v = np.zeros((n_fields, n_fields))
    
    for i in range(n_fields):
        for j in range(n_fields):
            if i != j:
                real_cramers_v[i, j] = cramers_v(real_df[fields[i]], real_df[fields[j]])
                synth_cramers_v[i, j] = cramers_v(synth_df[fields[i]], synth_df[fields[j]])
    
    # Создаем DataFrames
    real_cramers_v_df = pd.DataFrame(real_cramers_v, 
                                     index=fields, 
                                     columns=fields)
    synth_cramers_v_df = pd.DataFrame(synth_cramers_v, 
                                      index=fields, 
                                      columns=fields)
    
    return real_cramers_v_df, synth_cramers_v_df

def analyze_categorical_correlations(real_df, synth_df, categorical_fields=None):
    """
    Детальное сравнение корреляционных матриц между реальными и синтетическими данными
    """
    print(f"\n{'='*60}")
    print("CORRELATION ANALYSIS CATEGORICAL FIELDS")
    print(f"{'='*60}")
    
    results = {}
    
    
    if categorical_fields and len(categorical_fields) > 1:
        print("\n1. Categorical Feature")
        
        available_categorical = [f for f in categorical_fields if f in real_df.columns and f in synth_df.columns]
        
        if len(available_categorical) > 1:
            
  

            real_cramers_v_df, synth_cramers_v_df = count_cramers(real_df, synth_df, available_categorical)
            
            # Разница в ассоциациях
            cramers_v_diff = real_cramers_v_df - synth_cramers_v_df
            abs_cramers_v_diff = abs(cramers_v_diff)
            
            mask = np.triu(np.ones_like(real_cramers_v_df, dtype=bool), k=1)
            abs_diff_cramers_values = abs_cramers_v_diff.values[mask]
            
            results['categorical_associations'] = {
                'mean_absolute_difference': np.mean(abs_diff_cramers_values),
                'max_absolute_difference': np.max(abs_diff_cramers_values),
                'cramers_v_preservation_score': max(0, 1 - np.mean(abs_diff_cramers_values))
            }
            
            print(f"  Mean absolute difference (Cramér's V): {results['categorical_associations']['mean_absolute_difference']:.4f}")
            print(f"  Max absolute difference (Cramér's V): {results['categorical_associations']['max_absolute_difference']:.4f}")
            print(f"  Cramér's V preservation score: {results['categorical_associations']['cramers_v_preservation_score']:.3f}")
            
            # Визуализация Cramér's V
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            sns.heatmap(real_cramers_v_df, annot=True, fmt='.2f', cmap='YlOrRd', 
                       vmin=0, vmax=1, square=True, ax=axes[0])
            axes[0].set_title('Real Data: Cramér\'s V Associations')
            
            sns.heatmap(synth_cramers_v_df, annot=True, fmt='.2f', cmap='YlOrRd', 
                       vmin=0, vmax=1, square=True, ax=axes[1])
            axes[1].set_title('Synthetic Data: Cramér\'s V Associations')
            
            sns.heatmap(cramers_v_diff, annot=True, fmt='.2f', cmap='RdBu_r', 
                       center=0, square=True, ax=axes[2])
            axes[2].set_title('Cramér\'s V Difference (Real - Synthetic)')
            
            plt.tight_layout()
            plt.show()
    

def analyze_onehot_correlations(real_df, synth_df, onehot_fields):
    """
    Простой анализ корреляций между one-hot полями
    """
    print(f"\n{'='*60}")
    print("ONE-HOT FIELD CORRELATION ANALYSIS")
    print(f"{'='*60}")
    
    results = {}
    
    # 1. Кодируем one-hot поля для корреляционного анализа
    available_onehot = [f for f in onehot_fields if f in real_df.columns and f in synth_df.columns]
    
    if not available_onehot:
        print("No one-hot fields available for analysis")
        return results
    
    print(f"Available one-hot fields: {len(available_onehot)}")
    

    # 3. Создаем матрицы Cramér's V для реальных и синтетических данных
    print("\n1. Cramér's V Correlation Matrices for One-Hot Fields:")
    
    real_cramers_v_df, synth_cramers_v_df = count_cramers(real_df, synth_df, available_onehot)

    
    # Разница между матрицами
    cramers_v_diff = real_cramers_v_df - synth_cramers_v_df
    
    # 4. Визуализация матриц корреляций
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Real data heatmap
    sns.heatmap(real_cramers_v_df, annot=True, fmt='.2f', cmap='YlOrRd', 
                vmin=0, vmax=1, square=True, ax=axes[0], cbar_kws={'label': "Cramér's V"})
    axes[0].set_title('Real Data: Cramér\'s V between One-Hot Fields')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)
    
    # Synthetic data heatmap
    sns.heatmap(synth_cramers_v_df, annot=True, fmt='.2f', cmap='YlOrRd', 
                vmin=0, vmax=1, square=True, ax=axes[1], cbar_kws={'label': "Cramér's V"})
    axes[1].set_title('Synthetic Data: Cramér\'s V between One-Hot Fields')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)
    
    # Difference heatmap
    sns.heatmap(cramers_v_diff, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, square=True, ax=axes[2], cbar_kws={'label': 'Difference (Real - Synthetic)'})
    axes[2].set_title('Difference in Cramér\'s V (Real - Synthetic)')
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].tick_params(axis='y', rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    # 5. Статистики по различиям
    # Берем только верхний треугольник (без диагонали)
    mask = np.triu(np.ones_like(real_cramers_v_df, dtype=bool), k=1)
    diff_values = cramers_v_diff.values[mask]
    abs_diff_values = np.abs(diff_values)
    
    stats_dict = {
        'mean_absolute_difference': np.nanmean(abs_diff_values),
        'median_absolute_difference': np.nanmedian(abs_diff_values),
        'max_absolute_difference': np.nanmax(abs_diff_values),
        'std_absolute_difference': np.nanstd(abs_diff_values),
        'correlation_preservation_score': max(0, 1 - np.nanmean(abs_diff_values))
    }
    
    print(f"\n2. Correlation Difference Statistics:")
    print(f"   Mean absolute difference: {stats_dict['mean_absolute_difference']:.4f}")
    print(f"   Median absolute difference: {stats_dict['median_absolute_difference']:.4f}")
    print(f"   Max absolute difference: {stats_dict['max_absolute_difference']:.4f}")
    print(f"   Std of differences: {stats_dict['std_absolute_difference']:.4f}")
    print(f"   Correlation preservation score: {stats_dict['correlation_preservation_score']:.3f}")
    
    # 6. Топ-5 пар с наибольшими различиями
    print(f"\n3. Top 5 Pairs with Largest Differences:")
    n_fields = len(available_onehot)
    # Создаем список всех пар
    pairs_data = []
    for i in range(n_fields):
        for j in range(i+1, n_fields):  # Только верхний треугольник
            f1, f2 = available_onehot[i], available_onehot[j]
            real_val = real_cramers_v_df.loc[f1, f2]
            synth_val = synth_cramers_v_df.loc[f1, f2]
            diff = real_val - synth_val
            abs_diff = abs(diff)
            
            pairs_data.append({
                'field1': f1,
                'field2': f2,
                'real_cramers_v': real_val,
                'synth_cramers_v': synth_val,
                'difference': diff,
                'abs_difference': abs_diff
            })
    
    # Сортируем по абсолютной разнице
    pairs_df = pd.DataFrame(pairs_data)
    top_pairs = pairs_df.nlargest(5, 'abs_difference')
    
    for idx, row in top_pairs.iterrows():
        print(f"   {row['field1']} vs {row['field2']}:")
        print(f"     Real Cramér's V: {row['real_cramers_v']:.3f}")
        print(f"     Synthetic Cramér's V: {row['synth_cramers_v']:.3f}")
        print(f"     Difference: {row['difference']:.3f}")
    
    
def compare_correlation_between_num_fields_and_cat(real_data, synth_data, num_field, cat_field):

    if cat_field in real_data.columns and cat_field in synth_data.columns and num_field in real_data.columns and num_field in synth_data.columns:
        print(f"\n4. {num_field} vs {cat_field} Analysis:")
        
        # Для каждой валюты смотрим распределение курсов
        currencies = real_data[cat_field].unique()
        
        
        
        print(f"{num_field} Statistics by Currency:")
        print(f"{'Currency':<20} {'Real Mean':<12} {'Synth Mean':<12} {'Diff %':<10}")
        print("-" * 60)
        
        currency_stats = []
        for currency in currencies[:10]: 
            real_mean = real_data[real_data[cat_field] == currency][num_field].mean()
            synth_mean = synth_data[synth_data[cat_field] == currency][num_field].mean()
            
            if pd.notna(real_mean) and pd.notna(synth_mean) and real_mean != 0:
                diff_pct = abs(real_mean - synth_mean) / real_mean * 100
                currency_stats.append({
                    'currency': str(currency),
                    'real_mean': real_mean,
                    'synth_mean': synth_mean,
                    'diff_pct': diff_pct
                })
        
        currency_stats.sort(key=lambda x: x['diff_pct'], reverse=True)
        
        for stat in currency_stats[:5]:  
            print(f"{stat['currency']:<20} {stat['real_mean']:<12.4f} {stat['synth_mean']:<12.4f} {stat['diff_pct']:<10.1f}%")
