import os
import matplotlib.pyplot as plt

from data import load_jsonl

path = ''

path_info = os.path.dirname(path)
log_info = path.removeprefix(path_info)

logData = load_jsonl(path)


# plotting
train_step = []; train_loss = []
test_step = []; test_loss = []

for ind, line in enumerate(logData):
    if 'step' in line.keys():
        train_loss.append(line['step'])
        train_loss.append(line['train_loss'])
    else:
        test_step.append(logData[ind-1]['step'])
        test_loss.append(line['test_loss'])


X_tr = train_step; Y_tr = train_loss
X_ts = test_step; Y_ts = test_loss

plt.plot(X_tr, Y_tr, color='blue', label='Train_loss')
plt.plot(X_ts, Y_ts, color='green', label='Test_loss')
plt.title(f'{log_info}')
plt.xlabel('step'); plt.ylabel('loss')
plt.show()