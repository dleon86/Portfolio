# TestAccuracy.py
# compute test accuracy
import torch
from regAccuracy import regAcc

def TestAcc(model, test_loader, error, reg=False, th=1e-8,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    test_correct = testing_loss = batches = 0

    n_test = len(test_loader.sampler.indices)

    Yt_labs  = torch.empty(0, dtype=torch.float64, device=device)
    Yt_plabs = torch.empty(0, dtype=torch.float64, device=device)

    with torch.no_grad():
        model.eval()
        for Xt, Yt in test_loader:
            Xt, Yt = Xt.to(device), Yt.to(device)
            batches += 1

            Yt_pred = model(Xt)
            if reg == False:
                testing_loss += error(Yt_pred, Yt).item()
                # loss = error(Yt_pred, Yt)
                test_correct += (torch.argmax(Yt_pred, dim=1) == Yt).sum()
                # test_correct += (torch.argmax(Yt_pred.cpu(), dim=1) == Yt.cpu()).sum()
            else:
                Y_pred_class, test_correct = regAcc(Yt, Yt_pred, test_correct, th)   #train_correct
                testing_loss += error(Yt_pred, Yt.float()).item()
                # loss =  error(Yt_pred.squeeze(1), Yt.float())
            # testing_loss += loss.item()
            Yt_labs  = torch.cat((Yt_labs, Yt), dim=0)
            Yt_plabs = torch.cat((Yt_plabs, Y_pred_class), dim=0)

    test_accuracy = test_correct/n_test*100
    testing_loss /= batches
    if reg == False:
        print(f'Test set accuracy : {test_accuracy}')
    else:
        print(f'Test set MSE loss : {testing_loss}')
        print(f'Test set accuracy : {test_accuracy}')

    return test_accuracy, testing_loss, Yt_labs, Yt_plabs
