import torch
from torch.nn.modules.loss import _Loss
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

class Criterion(_Loss):
    def __init__(self, way=2, shot=5):
        super(Criterion, self).__init__()
        self.amount = way * shot

    def forward(self, probs, target):  # (Q,C) (Q)
        target = target[self.amount:]
        target_onehot = torch.zeros_like(probs)
        target_onehot = target_onehot.scatter(1, target.reshape(-1, 1), 1)
        loss = torch.mean((probs - target_onehot) ** 2)
        pred = torch.argmax(probs, dim=1)
        wp = precision_score(y_true = target,y_pred = pred, average='macro')
        wr = recall_score(y_true = target,y_pred = pred, average='macro')
        wf = f1_score(y_true = target,y_pred = pred, average='macro')
        log_record3 = open("log_100_precision.txt", mode = "a+", encoding = "utf-8")
        log_record4 = open("log_100_recall.txt", mode = "a+", encoding = "utf-8")
        log_record5 = open("log_100_f1-score.txt", mode = "a+", encoding = "utf-8")
        print("1"+" ,"+str(wp), file = log_record3)
        print("1"+" ,"+str(wr), file = log_record4)
        print("1"+" ,"+str(wf), file = log_record5)
        
        acc = torch.sum(target == pred).float() / target.shape[0]
        return loss, acc
