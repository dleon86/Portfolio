import os
import torch
import numpy as np
from tqdm import tqdm
from TestCVloss_plot import TestCVloss_plot
from regAccuracy import regAcc

def RDT_trainer(model, train_loader, val_loader,
                optimizer, error, lr_scheduler, tbwriter,
                train_loss, cv_loss, accuracy_list,
                NUM_EPOCHS, BATCH_SIZE, VAL_INTERVAL, CHECKPOINT_DIR,
                title, device, th, cont_train=False, reg=False):

# def RDT_trainer(pind, cont_train=False, reg=False):

    # set up counting variables that don't initialize if training is continuing
    if cont_train==False:
        # model = pind['model']
        # train_loader = pind['train_loader']
        # val_loader = pind['val_loader']
        # optimizer = pind['optimizer']
        # error = pind['error']
        # lr_scheduler = pind['lr_scheduler']
        # tbwriter = pind['tbwriter']
        # train_loss = pind['train_loss']
        # cv_loss = pind['cv_loss']
        # accuracy_list = pind['accuracy_list']
        # NUM_EPOCHS = pind['NUM_EPOCHS']
        # BATCH_SIZE = pind['BATCH_SIZE']
        # VAL_INTERVAL = pind['VAL_INTERVAL']
        # CHECKPOINT_DIR = pind['CHECKPOINT_DIR']
        # title = pind['title']
        # device = pind['device']

        # counters for averaging batch losses, among other things
        cont = total_steps = count1 = count3 = 0
        cv_loss_min = np.Inf  # Makes sure that we save the weights first time

    # Training RDTNet!!
    print('Starting Training...')
    total_steps += 1
    cont += 1
    for epoch in tqdm(range(NUM_EPOCHS), ascii=True):
        training_loss = cving_loss = correct_train = 0
        count2 = 0
        model.train()
        if epoch > 0:
            lr_scheduler.step()
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)   # adding 430 (itr2: additional 476) MB to cuda memory
            # break

            # calculate the loss
            # model.train()
            Y_pred = model(X)   # adding 1318 MB to mem
            if reg==False:
                loss = error(Y_pred, (Y))   # vrsion for class
            else:
                Y_pred_class, correct_train = regAcc(Y, Y_pred, correct_train, th)
                loss = error(Y_pred, Y.float()) # vrsn for regres
                # loss = error(Y_pred_class, Y.float()) # vrsn for regres
                # loss = error(Y_pred.squeeze(1), Y.float()) # vrsn for regres before accFnct
            # training_loss += loss   # .detach()  # for classification
            training_loss += loss.item()  # for regression

            # update the parameters
            # with torch.autograd.set_detect_anomaly(True):
            optimizer.zero_grad()  # adds to memory 262 Mb
            loss.backward()  # retain_graph=True)
            # loss.backward(retain_graph=False)   # adding 956 MB to mem

            # update weights
            optimizer.step()   # removing 466 MB

            count1 += 1
            count2 += 1

            # log the information and add to tensorboard
            if total_steps % 10 == 0:
                correct_train_spot = 0
                with torch.no_grad():
                    if reg == False:
                        preds = torch.argmax(Y_pred, dim=1)
                        # _, preds = torch.max(Y_pred)
                        # print('Ground truth', Y[0:24].cpu().detach(), '\nprediction : ',preds[0:24].cpu().detach())
                        accuracy = torch.sum(preds == Y)
                    else:
                        _, accuracy = regAcc(Y, Y_pred, correct_train_spot, th)

                    print('Epoch: {} \tStep: {} \tLoss: {:.5f} \tAcc: {} %'
                          .format(epoch + 1 + (cont - 1) * NUM_EPOCHS, total_steps, loss.item(),
                                  100 * accuracy.item() / BATCH_SIZE))  # epoch+1
                    tbwriter.add_scalar('loss', loss.item(), total_steps)
                    tbwriter.add_scalar('accuracy', accuracy.item() / BATCH_SIZE, total_steps)

            # print out gradient values and parameter average values
            if total_steps % 100 == 0:
                with torch.no_grad():
                    # print and save the grad of the parameters
                    # also print and save parameter values
                    print('*' * 50)
                    for name, parameter in model.named_parameters():
                        if parameter.grad is not None:
                            avg_grad = torch.mean(parameter.grad)
                            print('\t{} - grad_avg: {}'.format(name, avg_grad))
                            tbwriter.add_scalar('grad_avg/{}'.format(name), avg_grad.item(), total_steps)
                            tbwriter.add_histogram('grad/{}'.format(name),
                                                   parameter.grad.cpu().numpy(), total_steps)
                        if parameter.data is not None:
                            avg_weight = torch.mean(parameter.data)
                            print('\t{} - param_avg: {}'.format(name, avg_weight))
                            tbwriter.add_histogram('weight/{}'.format(name),
                                                   parameter.data.cpu().numpy(), total_steps)
                            tbwriter.add_scalar('weight_avg/{}'.format(name), avg_weight.item(), total_steps)

            total_steps += 1
            # lr_scheduler.step()

        training_loss = training_loss / len(train_loader.sampler)
        train_loss.append(training_loss)

        # start validation!!
        if (epoch + 1) % VAL_INTERVAL == 0:
            print('Starting Cross Validation...')
            total_Vsteps = 1
            CVcorrect = count4 = 0    # = tot
            with torch.no_grad():
                model.eval()
                for Xv, Yv in val_loader:
                    count3 += 1
                    count4 += 1
                    Xv, Yv = Xv.to(device), Yv.to(device)

                    # calculate the loss
                    Yv_pred = model(Xv)
                    if reg == False:  # for classification
                        CVloss = error(Yv_pred, Yv)
                        cving_loss += CVloss.item()
                        accuracy_list.append(100 * CVcorrect / len(Yv_pred))
                    else:   # for regression
                        Yv_pred_class, CVcorrect = regAcc(Yv, Yv_pred, CVcorrect, th)
                        CVloss = error(Yv_pred, Yv.float())  # vrsn for regres
                        cving_loss += CVloss.item()
                        accuracy_list.append(100 * CVcorrect.item() / len(Yv_pred))


                    # log the information and add to tensorboard
                    if total_Vsteps % 10 == 0:  # debugging
                        with torch.no_grad():
                            # _, CVpreds = torch.max(Yv_pred, 1)
                            # CVaccuracy = torch.sum(CVpreds == Yv)
                            if reg == False:
                                CVaccuracy = 100 * CVcorrect / len(Yv_pred)
                            else:
                                CVaccuracy = 100 * CVcorrect.item() / len(Yv_pred)

                            print('Epoch: {} \tCVStep: {} \tCVLoss: {:.4f} \tCVAcc: {} %'
                                  .format(epoch + 1 + (cont - 1) * NUM_EPOCHS, total_Vsteps, CVloss.item(),
                                          CVaccuracy))  # epoch+1
                                          # CVaccuracy.item() / BATCH_SIZE))  # epoch+1
                            tbwriter.add_scalar('CVloss', CVloss.item(), total_Vsteps)
                            tbwriter.add_scalar('CVaccuracy', CVaccuracy, total_Vsteps)
                            # tbwriter.add_scalar('CVaccuracy', CVaccuracy.item(), total_Vsteps)

                    total_Vsteps += 1

            ####
            cving_loss = cving_loss / len(val_loader.sampler)
            cv_loss.append(cving_loss)
            print(f"Validation Loss: {cving_loss: .5f}")
            print(f"Validation Accuracy: {CVcorrect / len(Yv_pred) * 100: .4f} %")  # epoch
            if epoch != 0:
                TestCVloss_plot(accuracy_list, train_loss, cv_loss, title + f'{epoch+1}')

        # save checkpoints
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'{title}_states.pkl')
        checkpoint_path2 = os.path.join(CHECKPOINT_DIR, f'{title}_state.pkl')
        state = {
            'epoch': epoch + 1 + (cont - 1) * NUM_EPOCHS,
            'total_steps': total_steps,
            'optimizer': optimizer.state_dict(),
            'model': model.state_dict(),
            # 'seed': seed,
        }
        if cving_loss <= cv_loss_min:
            print(f"Validation loss decreased({cv_loss_min: .6f} ==>{cving_loss: .6f}). Saving Model ...")
            torch.save(state, checkpoint_path)
            torch.save(model.state_dict(), checkpoint_path2)
            cv_loss_min = cving_loss
    return(model, train_loss, cv_loss, accuracy_list, Y, Y_pred, Yv, Yv_pred)

    # paramOutDict = {
    #     'model': model,
    #     'train_loss': train_loss,
    #     'cv_loss': cv_loss,
    #     'accuracy_list': accuracy_list,
    #     'Y': Y,
    #     'Y_pred': Y_pred,
    #     'Yv': Yv,
    #     'Yv_pred': Yv_pred,
    #     'cont': cont,
    #     'total_steps': total_steps,
    #     'count1': count1,
    #     'count3': count3,
    #     'cv_loss_min': cv_loss_min
    # }

