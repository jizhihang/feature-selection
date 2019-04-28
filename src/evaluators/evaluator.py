class Evaluator:
    def __init__(self):
        pass
    
    @staticmethod
    def eval_acc(pred_df):
        pred_df['correct_pred']=(pred_df['actual']==pred_df['pred'])
        acc = pred_df['correct_pred'].sum()/len(pred_df)
        return acc