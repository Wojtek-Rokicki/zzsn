

def log_train_metrics(args, losses, lr, epoch, wandb, verbose=True):
    if verbose:
        print(f"Epoch {epoch}: Train loss: {losses[0]:.5f} | Train Hungarian: {losses[1]:.5f} | " +
               f"Train KL: {losses[2]:.5f} | " +
              (f"Train N: {losses[3]:.5f} | " if losses[3] != 0 else '') +
              f"Train lr: {lr:.1e}")

    if args.wandb:
        dic = {name: losses[i] for i, name in enumerate(["Train loss", "Train Hungarian loss", "Train KL loss", "Train N loss"])}
        dic['lr'] = lr
        wandb.log(dic)


def log_test_metrics(args, losses, epoch, wandb, verbose=True):
    if verbose:
        print(f"Epoch {epoch}: Test loss: {losses[0]:.5f} | Test Hungarian: {losses[1]:.5f} | " +
              (f" Test N: {losses[2]:.5f} | " if losses[2] != 0 else ''))

    if args.wandb:
        dic = {name: losses[i] for i, name in enumerate(["Test loss", "Test Hungarian loss", "Test N loss"])}
        wandb.log(dic)


def log_evaluation_metrics(args, losses, epoch, wandb, extrapolation):
    ext = 'Extrapolation ' if extrapolation else ""
    print(f"Epoch {epoch}: {ext} Loss: {losses:.4f}")
    
    
    if args.wandb:
        wandb.log({f"{ext}Loss": losses})

