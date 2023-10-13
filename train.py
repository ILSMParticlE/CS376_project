import copy
import torch
import sys

WEIGHT_PATH = './weight/weight.pt'

def train_model(model, train_dataloader, valid_dataloader, train_len, valid_len,criterion, optimizer, num_epochs=25, is_inception=False, batch_size=64, weight_path=WEIGHT_PATH):
    ''' --- Initialization --- '''
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []
    val_loss_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(num_epochs):
        print('-' * 20)
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        train_loss = 0.0
        train_corrects = 0
        validation_loss = 0.0
        validation_corrects = 0

        ''' --- Train phase --- '''
        model.train()
        for X_train, y_train in train_dataloader:
            curr_batch_size = X_train.size(0)
            X_train = X_train.to(device)
            y_train = y_train.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            # train_loss += loss.item() * curr_batch_size
            # train_corrects += torch.sum(preds == y_train.data)
            train_loss += loss.detach() * curr_batch_size
            train_corrects += torch.sum(preds == y_train)
            del X_train
            del y_train


        epoch_train_loss = train_loss / train_len
        epoch_train_acc = train_corrects.double() / train_len
        train_acc_history.append(epoch_train_acc)
        train_loss_history.append(epoch_train_loss)

        print(f'Training Phase - Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')
        
        ''' --- Validation phase --- '''
        model.eval()
        for X_valid, y_valid in valid_dataloader:
            curr_batch_size = X_valid.size(0)
            X_valid = X_valid.to(device)
            y_valid = y_valid.to(device)

            with torch.no_grad():
                outputs = model(X_valid)
                loss = criterion(outputs, y_valid)

            preds = torch.argmax(outputs, dim=1)
            # validation_loss += loss.item() * curr_batch_size
            # validation_corrects += torch.sum(preds == y_valid.data)

            validation_loss += loss.detach() * curr_batch_size
            validation_corrects += torch.sum(preds == y_valid)
            del X_valid
            del y_valid

        epoch_validation_loss = validation_loss / valid_len
        epoch_validation_acc = validation_corrects.double() / valid_len
        val_acc_history.append(epoch_validation_acc)
        val_loss_history.append(epoch_validation_loss)
        print(f'Validation Phase - Loss: {epoch_validation_loss:.4f} Acc: {epoch_validation_acc:.4f}')

        if epoch_validation_acc > best_acc:
            print(f'Valid accuracy {epoch_validation_acc:.4f} is higher than best accuracy {best_acc:.4f} -> save weight')
            best_acc = epoch_validation_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, weight_path)

    torch.cuda.empty_cache()
    model.load_state_dict(best_model_wts)
    return model, train_acc_history, train_loss_history, val_acc_history, val_loss_history