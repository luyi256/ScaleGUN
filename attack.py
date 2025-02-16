import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch_geometric.seed import seed_everything
import numpy as np
from sklearn.model_selection import train_test_split
import copy


class AttackModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # inputs to hidden layer linear transformation
        # Note, when using Linear, weight and biases are randomly initialized for you
        self.hidden = nn.Linear(num_classes, 100)
        self.hidden2 = nn.Linear(100, 50)
        # output layer, 10 units - one for each digits
        self.output = nn.Linear(50, 2)

    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = F.sigmoid(self.hidden(x))
        x = F.sigmoid(self.hidden2(x))
        # output layer with softmax activation
        x = F.softmax(self.output(x), dim=1)
        # print("xxxxxxxx", x)

        return x


class Net(nn.Module):
    # define nn
    def __init__(self, num_classes):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(num_classes, 100)  # normal
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        # print("attack X",X)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        X = self.softmax(X)

        return X


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def attack_test(model, testloader, singleClass=False, trainTest=False):  # TODO N
    test_loss = 0
    test_accuracy = 0
    auroc = 0
    precision = 0
    recall = 0
    f_score = 0

    posteriors = []
    all_nodeIDs = []
    true_predicted_nodeIDs_and_class = {}
    false_predicted_nodeIDs_and_class = {}

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        # Doing validation

        # set model to evaluation mode
        model.eval()

        if trainTest:
            for features, labels in testloader:
                features, labels = features.to(
                    device), labels.to(device)
                # features = features.unsqueeze(1)  # unsqueeze
                features = features.view(features.shape[0], -1)
                logps = model(features)
                test_loss += criterion(logps, labels)

                # Actual probabilities
                ps = logps  # torch.exp(logps)
                posteriors.append(ps)

                # if singleclass=false
                # if not singleClass:
                #     y_true = labels.cpu().unsqueeze(-1)

                #     y_pred = ps.argmax(dim=-1, keepdim=True)

                #     # uncomment this to show AUROC
                #     auroc += roc_auc_score(y_true.cpu().numpy(),
                #                            y_pred.cpu().numpy())
                #     # print("auroc", auroc)

                #     precision += precision_score(
                #         y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                #     recall += recall_score(y_true.cpu().numpy(),
                #                            y_pred.cpu().numpy(), average='weighted')

                #     f_score += f1_score(y_true.cpu().numpy(),
                #                         y_pred.cpu().numpy(), average='weighted')

                top_p, top_class = ps.topk(1,
                                           dim=1)  # top_p gives the probabilities while top_class gives the predicted classes
                # print(top_p)
                equals = top_class == labels.view(
                    *top_class.shape)  # making the shape of the label and top class the same
                test_accuracy += torch.mean(
                    equals.type(torch.FloatTensor))

        else:
            print("len(testloader.dataset)", len(testloader.dataset))
            for features, labels, nodeIDs in testloader:
                print("nodeIDs", nodeIDs)
                features, labels = features.to(
                    device), labels.to(device)
                # features = features.unsqueeze(1)  # unsqueeze
                features = features.view(features.shape[0], -1)
                logps = model(features)
                test_loss += criterion(logps, labels)

                # Actual probabilities
                ps = logps  # torch.exp(logps)
                posteriors.append(ps)
                all_nodeIDs.append(nodeIDs)

                # if singleclass=false
                # if not singleClass:
                #     y_true = labels.cpu().unsqueeze(-1)
                #     # print("y_true", y_true)
                #     y_pred = ps.argmax(dim=-1, keepdim=True)
                #     # print("y_pred", y_pred)

                #     # uncomment this to show AUROC
                #     auroc += roc_auc_score(y_true.cpu().numpy(),
                #                            y_pred.cpu().numpy())
                #     # print("auroc", auroc)

                #     precision += precision_score(
                #         y_true.cpu().numpy(), y_pred.cpu().numpy(), average='weighted')

                #     recall += recall_score(y_true.cpu().numpy(),
                #                            y_pred.cpu().numpy(), average='weighted')

                #     f_score += f1_score(y_true.cpu().numpy(),
                #                         y_pred.cpu().numpy(), average='weighted')

                top_p, top_class = ps.topk(1,
                                           dim=1)
                equals = top_class == labels.view(
                    *top_class.shape)  # making the shape of the label and top class the same

                print("equals", len(equals))
                # for i in range(len(equals)):
                #     if equals[i]:

                #         true_predicted_nodeIDs_and_class[nodeIDs[i].item(
                #         )] = top_class[i].item()

                #     else:
                #         false_predicted_nodeIDs_and_class[nodeIDs[i].item(
                #         )] = top_class[i].item()

                test_accuracy += torch.mean(
                    equals.type(torch.FloatTensor))

    test_accuracy = test_accuracy / len(testloader)
    test_loss = test_loss / len(testloader)
    final_auroc = auroc / len(testloader)
    final_precision = precision / len(testloader)
    final_recall = recall / len(testloader)
    final_f_score = f_score / len(testloader)

    return test_loss, test_accuracy, posteriors, final_auroc, final_precision, final_recall, final_f_score, true_predicted_nodeIDs_and_class, false_predicted_nodeIDs_and_class


def attack_train(model, trainloader, testloader, criterion, optimizer, epochs, steps=0):
    # train ntwk
    final_train_loss = 0
    train_losses, test_losses = [], []
    best_test = -1
    pat = 1
    posteriors = []
    for e in range(epochs):
        running_loss = 0
        train_accuracy = 0

        for features, labels in trainloader:
            model.train()
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()

            features = features.view(features.shape[0], -1)

            logps = model(features)  # log probabilities
            # print("labelsssss", labels.shape)
            loss = criterion(logps, labels)

            # Actual probabilities
            # torch.exp(logps) #Only use this if the loss is nlloss
            ps = logps

            top_p, top_class = ps.topk(1,
                                       dim=1)  # top_p gives the probabilities while top_class gives the predicted classes
            # print(top_p)
            equals = top_class == labels.view(
                *top_class.shape)  # making the shape of the label and top class the same
            train_accuracy += torch.mean(
                equals.type(torch.FloatTensor))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:

            test_loss, test_accuracy, _, _, _, _, _, _, _ = attack_test(
                model, testloader, trainTest=True)

            # set model back yo train model
            model.train()
            # scheduler.step()

            train_losses.append(running_loss / len(trainloader))
            test_losses.append(test_loss)

            # get final train loss. To be returned at the end of the training loop
            final_train_loss = running_loss / len(trainloader)

            print("Epoch: {}/{}..".format(e + 1, epochs),
                  "Training loss: {:.5f}..".format(
                      running_loss / len(trainloader)),
                  "Test Loss: {:.5f}..".format(test_loss),
                  "Train Accuracy: {:.3f}".format(
                      train_accuracy / len(trainloader)),
                  "Test Accuracy: {:.3f}".format(test_accuracy)
                  )
            if test_accuracy > best_test:
                best_test = test_accuracy
                best_model = copy.deepcopy(model)
                pat = 1
            else:
                pat += 1
                if pat > 10:
                    break

    # # plot train and test loss
    # plt.show()
    # plt.plot(train_losses)
    # plt.plot(test_losses)
    # plt.title('Model Losses')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()

    return final_train_loss, best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ogbn-products")
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--dev", type=int, default=1)
    parser.add_argument("--num_classes", type=int, default=47)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    seeds = [1050154401, 87952126, 461858464, 2251922041,
             2203565404, 2569991973, 569824674, 2721098863, 836273002, 2935227127]
    device = torch.device(f"cuda:{args.dev}")
    for run in range(1):
        seed = seeds[args.seed]
        seed_everything(seeds[args.seed])
        attack_model = Net(args.num_classes)  # AttackModel()
        attack_model = attack_model.to(device)
        criterion = nn.CrossEntropyLoss()  # nn.NLLLoss() # cross entropy loss
        optimizer = torch.optim.Adam(
            attack_model.parameters(), lr=0.001)  # 0.01 #0.00001
        train_x = np.load(
            f'./analysis/{args.dataset}/node_feat_retrain_test_MI/MI_pred_train_logits_{args.seed}.npy')
        train_y = np.ones(train_x.shape[0], dtype=np.int64)
        test_x = np.load(
            f'./analysis/{args.dataset}/node_feat_retrain_test_MI/MI_pred_test_logits_{args.seed}.npy')
        test_y = np.zeros(test_x.shape[0], dtype=np.int64)
        attack_x = np.concatenate([train_x, test_x])
        attack_y = np.concatenate([train_y, test_y])
        attack_x_train, attack_x_test, attack_y_train, attack_y_test = train_test_split(
            attack_x, attack_y, test_size=0.1, stratify=attack_y, random_state=seed)
        attack_train_data = torch.utils.data.TensorDataset(torch.from_numpy(attack_x_train).float(), torch.from_numpy(
            attack_y_train))  # convert to float to fix  uint8_t overflow error
        attack_train_data_loader = torch.utils.data.DataLoader(
            attack_train_data, batch_size=32, shuffle=True)

        # Attack_test = combo of targettrain and targetOut
        attack_test_data = torch.utils.data.TensorDataset(torch.from_numpy(attack_x_test).float(), torch.from_numpy(
            attack_y_test))  # convert to float to fix  uint8_t overflow error
        attack_test_data_loader = torch.utils.data.DataLoader(
            attack_test_data, batch_size=1024, shuffle=True)  # 1000
        _, attack_model = attack_train(attack_model, attack_train_data_loader,
                                       attack_test_data_loader, criterion, optimizer, args.epochs)
        save_attack_model = f'./models/attack_model_{args.dataset}_{args.seed}.pt'
        torch.save(attack_model.state_dict(), save_attack_model)
