import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_selection(feature_df, feature_set_name, show_features=False):
    feature_df = feature_df[feature_df['continue']==True]
    feature_melt_df = pd.melt(feature_df, id_vars=['features','num_features'],value_vars=['test_acc','val_acc'])
    feature_melt_df = feature_melt_df.rename(columns={'variable':'Set', 'value':'Accuracy', 'features':'Features', 'num_features':'Number of Features'})
    feature_melt_df['Set'] = feature_melt_df['Set'].str.split('_').map(lambda x: x[0])
    feature_melt_df['Features'] = feature_melt_df['Features'].map(lambda x: '['+','.join(x)+']')
    feature_melt_df['Error'] = 100*(1-feature_melt_df['Accuracy'])
    
    fig,ax = plt.subplots(1,1,figsize=(15,10))
    if show_features:
        plt.xticks(rotation=90)
        sns.barplot(x='Features', y='Error', hue='Set', data=feature_melt_df)
        ax.set_title('Error vs Features for the {} Data Set'.format(feature_set_name))
    else:
        sns.barplot(x='Number of Features', y='Error', hue='Set', data=feature_melt_df)
        ax.set_title('Error vs Number of Features for the {} Data Set'.format(feature_set_name))
