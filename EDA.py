# Downloaded Packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelBinarizer


class EDA:

    def __init__(self, name, df):
        self.df = df
        self.name = name

        mpl.rcParams['agg.path.chunksize'] = 10000

    def missing(self):
        for column in self.df.columns:
            print(f'% Missing Values from {column}: {self.df[column].isna().sum()}')
        print('\n')

    def hists(self):
        pdf = matplotlib.backends.backend_pdf.PdfPages(
            f"EDA_{self.name}.pdf")
        for i, col in enumerate(self.df.select_dtypes(include=['float64', 'int64'])):
            if self.df[col].isna().sum() == 0:
                plt.figure(i)
                print(f'Median {col}: {np.median(self.df[col]).round(2)}')
                print(f"Mean {col}: {np.mean(self.df[col]).round(2)}")
                print(f"STD {col}: {np.std(self.df[col]).round(2)}\n")
                sns.distplot(self.df[col])
                pdf.savefig()
        pdf.close()

    def binarate(self, target_var):
        pdf = matplotlib.backends.backend_pdf.PdfPages(
            f"EDA_{self.name}_1.pdf")
        for i, col in enumerate(self.df.select_dtypes(include=['float64', 'int64']).drop(target_var, axis=1)):
            if self.df[col].isna().sum() == 0:
                plt.figure()
                plt.scatter(self.df[col], self.df[target_var])
                plt.xlabel(col)
                plt.ylabel(target_var)
                pdf.savefig()
        pdf.close()
        
    def heatmap(self, metric):
        if metric == 'corr':
            x = self.df.corr()
            x.to_csv(f'{self.name}_corr_matrix.csv')
        elif metric == 'cov':
            x = self.df.cov()
            x.to_csv(f'{self.name}_cov_matrix.csv')
        else:
            raise ValueError(
                "Invalid argument: select 'corr'(elation) or 'cov'(ariance)")
        sns.heatmap(x)
        plt.show()

    def categorical(self, categories, target_var):
        pdf = matplotlib.backends.backend_pdf.PdfPages(
            f"EDA_{self.name}_2.pdf")
        y = self.df[target_var]
        X = self.df.drop(target_var, axis=1)
        for i, cat in enumerate(categories):
            plt.clf()
            # plt.figure(i)
            mapper = DataFrameMapper([
                (cat, LabelBinarizer())])
            new = pd.DataFrame(mapper.fit_transform(X.copy()),
                               columns=mapper.transformed_names_)
            values = []
            counts = []
            for new_col in new.columns:
                sub = new.index[new[new_col] == 1].tolist()
                counts.append(len(y[sub]))
                values.append((y[sub].sum())/len(y[sub]))
            keys = [(x.replace(cat + '_', '') + ':\n' + str(counts.pop()))
                    for x in new.columns]
            pairs = list(zip(keys, values))
            sorted(pairs, key=lambda x: x[0])
            keys = [x[0] for x in pairs]
            values = [x[1] for x in pairs]
            data = pd.DataFrame({cat: keys, target_var: values})
            plot = sns.barplot(x=cat, y=target_var, data=data)
            rotation = 5*len(keys)
            if rotation > 90:
                rotation = 90
            plot.set_xticklabels(plot.get_xticklabels(),
                                 rotation=rotation)
            plt.suptitle(f"{cat} vs. {target_var}")
            plt.tight_layout()
            pdf.savefig()
        pdf.close()

    def feature_relationships(self, which, target_var=None, top=None, order=None):
        if which == 'corr':
            x = self.df.corr()
            x.to_csv(f'{self.name}_corr_matrix.csv')
        elif which == 'cov':
            x = self.df.cov()
            x.to_csv(f'{self.name}_cov_matrix.csv')
        else:
            raise ValueError(
                "Invalid argument: select 'corr'(elation) or 'cov'(ariance)")
        if target_var:
            if order == None:  # sort by descending absolute value
                x = x.reindex(
                    x[target_var].abs().sort_values().index)
                # skip first term because will be 1.00 from correlation with self
                x = x[::-1][target_var][1:]
            elif order == 'desc':
                x = x[target_var].sort_values(ascending=False)
                # skip first term because will be 1.00 from correlation with self
                x = x[1:]
            elif order == 'asc':
                x = x[target_var].sort_values(ascending=True)
            else:
                raise ValueError(
                    "Invalid order: select 'asc'(ending) or 'desc'(ending). Default is to sort desc by absolute value")
            if top:
                return(x[:top])
            else:
                return(x)
        else:
            if top:
                raise ValueError(
                    "Cannot pass 'top' argument without a target variable")
            if order:
                raise ValueError(
                    "Cannot pass 'order' argument without a target variable")
            print('No target variable passed, returning entire matrix')
            return(x)

    def multivariate_plots(self, cat_var, numeric_var, target_var):
        y = self.df[target_var]
        X = self.df.drop(target_var, axis=1)
        pdf = matplotlib.backends.backend_pdf.PdfPages(
            f"EDA_{self.name}_3.pdf")

        cols = []
        for col in X.columns:
            if cat_var in col:
                cols.append(col)
        if len(cols) > 1:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.spines['top'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.tick_params(labelcolor='w', top='off',
                           bottom='off', left='off', right='off')
            col = cols[0]
            subset = X.loc[(X[col] == 1) & (X[numeric_var] != 0)]
            sub_y = y.loc[(X[col] == 1) & (X[numeric_var] != 0)]
            subset[target_var] = sub_y
            ax1 = fig.add_subplot(len(cols), 1, 1)
            ax1.set_title(col.replace(cat_var + '_', ''))
            sns.regplot(x=numeric_var, y=target_var, data=subset)
            ax1.set_xlabel('')
            for i, col in enumerate(cols[1:]):
                subset = X.loc[(X[col] == 1) & (X[numeric_var] != 0)]
                sub_y = y.loc[(X[col] == 1) & (X[numeric_var] != 0)]
                subset[target_var] = sub_y
                ax_ = fig.add_subplot(len(cols), 1, i+2, sharex=ax1)
                ax_.set_title(col.replace(cat_var + '_', ''))
                sns.regplot(x=numeric_var, y=target_var, data=subset)
                ax_.set_xlabel('')
            ax.set_xlabel(numeric_var)
            plt.suptitle(cat_var)
            plt.subplots_adjust(hspace=.5)
            plt.show()
        else:
            # means the categorical data in a binary yes or no in one column
            yes = X[X[cat_var] == 1]
            y_yes = y[X[cat_var] == 1]
            yes[target_var] = y
            no = X[X[cat_var] == 0]
            y_no = y[X[cat_var] == 0]
            no[target_var] = y
            plt.subplot(2, 1, 1)
            sns.scatterplot(x=numeric_var, y=target_var, data=yes)
            plt.subplot(2, 1, 2)
            sns.scatterplot(x=numeric_var, y=target_var, data=no)
            plt.tight_layout()
            plt.show()
        plt.close()
